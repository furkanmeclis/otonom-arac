"""Hedef nokta -> direksiyon/gaz dönüşümü (geometrik kontrolcü).

Bu modül projenin "karar verme" katmanıdır. CNN modeli sadece bir
hedef nokta (target_x, target_y) tahmin eder; aracı nasıl süreceğini
BURASI hesaplar. Yani model "nereye gitmeliyim?" der, bu modül "o hâlde
direksiyonu şu kadar çevir, gazı şu kadar bas" der.

Akıştaki yeri:
    kamera -> CNN modeli -> (target_x, target_y) -> [BU MODÜL] -> direksiyon, gaz

İki parça vardır:
  * target_point_to_controls(): tek karelik (anlık) saf geometrik hesap.
    Hedef noktanın açısından (atan2) direksiyonu, eğrilikten gazı üretir.
  * TargetPointController: DonkeyCar "part"ı. Yukarıdaki saf hesabın
    üstüne zaman-bilgisi ekler (öngörülü gaz + direksiyon yumuşatma).

Tüm uzunluklar metre, açılar radyan; ego-frame (aracın kendi koordinatı):
  +y ileri, +x sağ. heading_error = atan2(target_x, target_y).
"""

from __future__ import annotations

import math
from typing import Tuple


def _is_valid_number(value) -> bool:
    """Sayı geçerli mi? (None, NaN, sonsuz değilse True). Model bazen
    bozuk değer üretebilir; bunları kontrolcüye sokmadan eleriz."""
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def target_point_to_controls(
    target_x: float,
    target_y: float,
    steer_gain: float,
    steer_sign: float,
    throttle: float,
    min_forward: float,
    curvature_gain: float = 0.0,
    angle_boost: float = 0.0,
    lateral_boost: float = 0.0,
    recovery_angle_deg: float = 10.0,
    recovery_target_x_m: float = 0.2,
    turn_throttle_reduction: float = 0.0,
    min_throttle_scale: float = 1.0,
    dynamic_throttle: bool = False,
    base_throttle: float = 0.35,
    min_throttle: float = 0.12,
    curvature_throttle_angle_deg: float = 15.0,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """Bir hedef noktayı (target_x, target_y) direksiyon ve gaza çevirir.

    Girdi (önemliler):
        target_x, target_y : hedef nokta, ego-frame metre (y ileri, x sağ).
        steer_gain         : yön hatasını direksiyona çevirirken kazanç.
        steer_sign         : montaj yönüne göre direksiyonu ters çevirmek (+1/-1).
        dynamic_throttle   : True ise gaz, viraj keskinliğine göre sürekli
                             ayarlanır (düzde base_throttle, virajda min_throttle).
        recovery_*         : araç çok saparsa devreye giren ek kazanç eşikleri.

    Döner:
        (steering, throttle) : ikisi de normalize, direksiyon [-1, 1].

    Mantık özet:
        heading_error = atan2(target_x, target_y)   # hedefe olan açı
        steering      = tanh(steer_gain * heading_error)  # tanh ile [-1,1]'e sıkıştır
    Geçersiz/çok yakın hedefte (target_y < min_forward) araç durur (0, 0).
    """
    if not _is_valid_number(target_x) or not _is_valid_number(target_y):
        return 0.0, 0.0

    target_x = float(target_x)
    target_y = float(target_y)
    if target_y < float(min_forward):
        return 0.0, 0.0

    heading_error = math.atan2(target_x, max(target_y, eps))
    curvature = 2.0 * math.sin(heading_error) / max(target_y, 0.15)

    recovery_angle_rad = math.radians(max(float(recovery_angle_deg), 1e-3))
    angle_activation = 0.0
    if abs(heading_error) > recovery_angle_rad:
        angle_activation = min(1.0, (abs(heading_error) - recovery_angle_rad) / recovery_angle_rad)

    lateral_threshold = max(float(recovery_target_x_m), 1e-3)
    lateral_activation = 0.0
    if abs(target_x) > lateral_threshold:
        lateral_activation = min(1.0, (abs(target_x) - lateral_threshold) / lateral_threshold)

    gain_scale = 1.0 + float(angle_boost) * angle_activation + float(lateral_boost) * lateral_activation
    curvature_activation = max(angle_activation, lateral_activation)
    command = gain_scale * (
        float(steer_gain) * heading_error + float(curvature_gain) * curvature_activation * curvature
    )
    steering = float(steer_sign) * math.tanh(command)
    steering = max(-1.0, min(1.0, steering))

    if dynamic_throttle:
        max_angle_rad = math.radians(max(float(curvature_throttle_angle_deg), 1e-3))
        curvature_score = min(1.0, abs(heading_error) / max_angle_rad)
        throttle_out = float(base_throttle) - (float(base_throttle) - float(min_throttle)) * curvature_score
    else:
        throttle_scale = 1.0 - float(turn_throttle_reduction) * curvature_activation
        throttle_scale = max(float(min_throttle_scale), throttle_scale)
        throttle_out = float(throttle) * throttle_scale

    return steering, throttle_out


class TargetPointController:
    """Tahmin edilen hedef noktayı sürüş komutuna çeviren DonkeyCar part'ı.

    Üstteki saf geometrik hesabın (target_point_to_controls) üstüne
    ZAMAN bilgisi ekler. Sürüş sırasında her karede run() çağrılır.

    Eklediği iki akıllı davranış:

    1. Öngörülü gaz (anticipatory throttle): Son birkaç karenin yön-hatasını
       hafızada tutar. Hata ARTIYORsa (viraj yaklaşıyor demektir) gazı erken
       keser, yani araç viraja girmeden yavaşlamaya başlar. Direksiyon ise
       anlık kalır ki takip hassasiyeti bozulmasın.

    2. Direksiyon hız sınırı (rate limit): Kareden kareye direksiyonun en
       fazla ne kadar değişebileceğini sınırlar. Ani sarsıntıları önler ama
       alçak-geçiren filtre gibi gecikme yaratmaz.

    Tüm eşikler/kazançlar cfg (simulationconfig.py) içinden okunur; böylece
    kodu değiştirmeden config'ten ayarlanabilir.
    """

    def __init__(self, cfg) -> None:
        self.steer_gain = float(getattr(cfg, "TARGET_POINT_STEER_GAIN", 1.0))
        self.steer_sign = float(getattr(cfg, "TARGET_POINT_STEER_SIGN", 1.0))
        self.throttle = float(getattr(cfg, "TARGET_POINT_THROTTLE", 0.2))
        self.min_forward = float(getattr(cfg, "TARGET_POINT_MIN_FORWARD", 0.0))
        self.curvature_gain = float(getattr(cfg, "TARGET_POINT_CURVATURE_GAIN", 0.0))
        self.angle_boost = float(getattr(cfg, "TARGET_POINT_ANGLE_BOOST", 0.0))
        self.lateral_boost = float(getattr(cfg, "TARGET_POINT_LATERAL_BOOST", 0.0))
        self.recovery_angle_deg = float(getattr(cfg, "TARGET_POINT_RECOVERY_ANGLE_DEG", 10.0))
        self.recovery_target_x_m = float(getattr(cfg, "TARGET_POINT_RECOVERY_TARGET_X_M", 0.2))
        self.turn_throttle_reduction = float(getattr(cfg, "TARGET_POINT_TURN_THROTTLE_REDUCTION", 0.0))
        self.min_throttle_scale = float(getattr(cfg, "TARGET_POINT_MIN_THROTTLE_SCALE", 1.0))
        self.dynamic_throttle = bool(getattr(cfg, "TARGET_POINT_DYNAMIC_THROTTLE", False))
        self.base_throttle = float(getattr(cfg, "TARGET_POINT_BASE_THROTTLE", 0.35))
        self.min_throttle = float(getattr(cfg, "TARGET_POINT_MIN_THROTTLE", 0.12))
        self.curvature_throttle_angle_deg = float(getattr(cfg, "TARGET_POINT_CURVATURE_THROTTLE_ANGLE_DEG", 15.0))
        # Runtime lateral-bias compensation for systematic right/left drift.
        # Effective target_x used by controller = raw_target_x - target_x_bias_m.
        self.target_x_bias_m = float(getattr(cfg, "TARGET_POINT_TARGET_X_BIAS_M", 0.0))
        # Ignore tiny residual target_x values around center to reduce steering jitter.
        self.target_x_deadband_m = float(getattr(cfg, "TARGET_POINT_TARGET_X_DEADBAND_M", 0.0))
        # Steering rate limit: max steering change per frame (0 = off)
        self.steer_rate_limit = float(getattr(cfg, "TARGET_POINT_STEER_RATE_LIMIT", 0.0))
        # Anticipation: how many frames of heading-error history to keep
        self.anticipation_frames = int(getattr(cfg, "TARGET_POINT_ANTICIPATION_FRAMES", 10))
        # Anticipation gain: how much the rising trend boosts effective curvature
        self.anticipation_gain = float(getattr(cfg, "TARGET_POINT_ANTICIPATION_GAIN", 2.0))

        self._prev_steering: float | None = None
        self._heading_history: list[float] = []

    def run(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """Her karede çağrılır: hedef noktayı (steering, throttle) yapar.

        Adımlar: (1) bias/deadband ile ham target_x'i düzelt,
        (2) geometrik kontrolcüden temel direksiyonu al,
        (3) yön-hatası geçmişinden öngörülü gazı hesapla,
        (4) direksiyonu hız sınırından geçir. Döner: (direksiyon, gaz)."""
        tx_raw = float(target_x) if _is_valid_number(target_x) else 0.0
        tx_adjusted = tx_raw - self.target_x_bias_m
        if self.target_x_deadband_m > 0.0 and abs(tx_adjusted) < self.target_x_deadband_m:
            tx_adjusted = 0.0

        # --- Get base steering from the geometric controller ---
        # Pass dynamic_throttle=False so we compute throttle ourselves
        # with anticipation below.
        steering, _base_throttle_out = target_point_to_controls(
            target_x=tx_adjusted,
            target_y=target_y,
            steer_gain=self.steer_gain,
            steer_sign=self.steer_sign,
            throttle=self.throttle,
            min_forward=self.min_forward,
            curvature_gain=self.curvature_gain,
            angle_boost=self.angle_boost,
            lateral_boost=self.lateral_boost,
            recovery_angle_deg=self.recovery_angle_deg,
            recovery_target_x_m=self.recovery_target_x_m,
            turn_throttle_reduction=self.turn_throttle_reduction,
            min_throttle_scale=self.min_throttle_scale,
            dynamic_throttle=False,
            base_throttle=self.base_throttle,
            min_throttle=self.min_throttle,
            curvature_throttle_angle_deg=self.curvature_throttle_angle_deg,
        )

        # --- Compute heading error for history tracking ---
        tx = tx_adjusted
        ty = float(target_y) if _is_valid_number(target_y) else 0.0
        if ty > float(self.min_forward):
            heading_error = math.atan2(tx, max(ty, 1e-6))
        else:
            heading_error = 0.0

        # --- Anticipatory throttle ---
        self._heading_history.append(abs(heading_error))
        if len(self._heading_history) > self.anticipation_frames:
            self._heading_history.pop(0)

        if self.dynamic_throttle:
            current_curvature = abs(heading_error)

            # Detect trend: compare recent half vs older half of history
            if len(self._heading_history) >= 4:
                half = len(self._heading_history) // 2
                recent_avg = sum(self._heading_history[half:]) / len(self._heading_history[half:])
                older_avg = sum(self._heading_history[:half]) / half
                trend = recent_avg - older_avg  # positive = curve intensifying

                # Boost effective curvature when trend is rising
                anticipated = current_curvature + max(0.0, trend) * self.anticipation_gain
                effective_curvature = max(current_curvature, anticipated)
            else:
                effective_curvature = current_curvature

            max_angle_rad = math.radians(max(self.curvature_throttle_angle_deg, 1e-3))
            curvature_score = min(1.0, effective_curvature / max_angle_rad)
            throttle_out = self.base_throttle - (self.base_throttle - self.min_throttle) * curvature_score
        else:
            throttle_out = _base_throttle_out

        # --- Rate-limit: cap frame-to-frame steering jumps ---
        if self.steer_rate_limit > 0.0 and self._prev_steering is not None:
            delta = steering - self._prev_steering
            if abs(delta) > self.steer_rate_limit:
                steering = self._prev_steering + math.copysign(self.steer_rate_limit, delta)
        self._prev_steering = steering

        return steering, throttle_out
