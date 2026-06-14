"""Teacher (öğretmen) politikaları ve hedef-nokta etiketleme kuralları.

Bu modül, SİMÜLASYONDA "doğru cevabı" üreten beyindir. CNN modeli daha
yokken, simülatörün bize bedava verdiği gerçek konum (pose) ve pist merkez
çizgisini kullanarak her kare için ideal (target_x, target_y) etiketini
hesaplar. Model sonra bu etiketleri taklit etmeyi öğrenir.

İçindeki üç ana iş:

  1. Projeksiyon: project_pose_to_centerline()
     Aracın o anki konumunu pist merkez çizgisine düşürür (en yakın nokta),
     yol üzerindeki ilerleme mesafesini, eğriliği, referans hızı bulur.

  2. Etiketleme: materialize_label_from_state() / materialize_label_mode()
     Merkez çizgisi boyunca "lookahead" kadar ileri bakar, o ileri noktayı
     aracın ego-frame'ine çevirir = (target_x, target_y). lookahead, hız ve
     viraj keskinliğine göre değişir (fixed_1p2m veya adaptive_v1 modu).

  3. Sürücü teacher'ları (veri toplarken simülatörü kim sürer?):
     - CleanMappingTeacher: pisti temiz/ortadan sürerek HARİTA çıkarır (Faz 1).
     - LowNoiseTargetPointTeacher: kasıtlı gürültü, şerit sapması ve
       kurtarma (recovery) manevralarıyla sürer (Faz 2/3). Böylece model
       sadece mükemmel sürüşü değil, hatadan dönmeyi de öğrenir.

Koordinatlar: dünya ekseni (x, z yatay düzlem) ve ego-frame (y ileri, x sağ).
Bu modül GERÇEK araçta değil, yalnızca etiket üretiminde kullanılır.
"""

from __future__ import annotations

from collections import deque
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from target_point.sim_session import DonkeySimSession, SimObservation
from target_point.track_map import CenterlineWaypoint, TrackMapArtifact, interpolate_centerline_position


FIXED_1P2M = "fixed_1p2m"
ADAPTIVE_V1 = "adaptive_v1"
COLLECTION_PROFILE_PHASE2 = "phase2_low_noise"
COLLECTION_PROFILE_PHASE3 = "phase3_full_noise"
LOOKAHEAD_SPEED_BINS = (0.0, 1.0, 2.0, 3.0, float("inf"))
LOOKAHEAD_CURVATURE_BINS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class TeacherAction:
    steering: float
    throttle: float
    brake: float


@dataclass(frozen=True)
class LabelRecord:
    map_id: str
    track_name: str
    label_mode: str
    waypoint_index: int
    distance_m: float
    reference_speed_mps: float
    curvature_radpm: float
    delta_heading_2m_deg: float
    curvature_score: float
    lookahead_m: float
    target_x: float
    target_y: float


@dataclass(frozen=True)
class ProjectedTrackState:
    map_id: str
    track_name: str
    map_distance_m: float
    waypoint_index: int
    next_waypoint_index: int
    interpolation_alpha: float
    pos_x: float
    pos_y: float
    pos_z: float
    heading_x: float
    heading_z: float
    yaw_deg: float
    distance_to_centerline_m: float
    curvature_radpm: float
    delta_heading_2m_deg: float
    curvature_score: float
    reference_speed_mps: float


@dataclass(frozen=True)
class StateLabel:
    label_mode: str
    lookahead_m: float
    target_x: float
    target_y: float
    tail_clamped: bool


@dataclass(frozen=True)
class LowNoiseTeacherOutput:
    action: TeacherAction
    projected_state: ProjectedTrackState
    clean_label: StateLabel
    collection_profile: str
    noise_target_x: float
    noise_target_y: float
    lane_bias_offset_x: float
    steering_noise: float
    throttle_noise: float
    teacher_commanded_steering: float
    teacher_commanded_throttle: float
    steering_lag_frames: int
    safety_override_active: bool
    recovery_steering_bias: float
    recovery_target_offset_x: float
    recovery_throttle_scale: float
    applied_target_x: float
    applied_target_y: float
    applied_lookahead_m: float
    scenario: str
    disturbance_type: str


def _clip(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _blend_with_rate_limit(previous: Optional[float], target: float, smoothing: float, max_delta: float) -> float:
    target = float(target)
    if previous is None:
        return target

    smoothing = _clip(float(smoothing), 0.0, 0.99)
    blended = smoothing * float(previous) + (1.0 - smoothing) * target
    if float(max_delta) > 0.0:
        blended = float(previous) + _clip(blended - float(previous), -float(max_delta), float(max_delta))
    return float(blended)


def _wrap_delta_deg(current: float, previous: float) -> float:
    return ((float(current) - float(previous) + 180.0) % 360.0) - 180.0


def _heading_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    yaw_rad = math.radians(float(yaw_deg))
    return np.asarray([math.sin(yaw_rad), math.cos(yaw_rad)], dtype=np.float32)


def _world_to_ego_from_pose(
    current_position_xyz: Sequence[float],
    yaw_deg: float,
    target_position_xyz: np.ndarray,
) -> tuple[float, float]:
    current = np.asarray([float(current_position_xyz[0]), float(current_position_xyz[2])], dtype=np.float32)
    heading = _heading_from_yaw_deg(float(yaw_deg))
    right = np.asarray([heading[1], -heading[0]], dtype=np.float32)
    delta = np.asarray(target_position_xyz[[0, 2]], dtype=np.float32) - current
    return float(np.dot(delta, right)), float(np.dot(delta, heading))


def _interpolate_scalar(start: float, end: float, alpha: float) -> float:
    return float(start) + (float(end) - float(start)) * float(alpha)


def _centerline_segment_count(track_map: TrackMapArtifact) -> int:
    if track_map.is_closed:
        return len(track_map.centerline)
    return max(len(track_map.centerline) - 1, 0)


def project_pose_to_centerline(
    track_map: TrackMapArtifact,
    pos_x: float,
    pos_y: float,
    pos_z: float,
) -> ProjectedTrackState:
    """Aracın (pos_x, pos_z) konumunu pist merkez çizgisine projekte eder.

    Tüm merkez-çizgi segmentlerini gezip aracın en yakın olduğu noktayı bulur,
    o noktadaki yol-ilerlemesini (map_distance_m), yönü, eğriliği ve referans
    hızı (segment iki ucu arasında doğrusal) interpolasyonla hesaplar.
    Etiketlemenin ilk adımı; 'şu an yolun neresindeyim?' sorusunu yanıtlar.
    Döner: ProjectedTrackState (projekte konum + yol metrikleri)."""
    best = None
    point = np.asarray([float(pos_x), float(pos_z)], dtype=np.float32)
    segment_count = _centerline_segment_count(track_map)
    if segment_count <= 0:
        raise ValueError(f"Track map {track_map.map_id} does not have enough waypoints for projection.")

    for waypoint_index in range(segment_count):
        current = track_map.centerline[waypoint_index]
        next_waypoint = track_map.centerline[(waypoint_index + 1) % len(track_map.centerline)]
        start = np.asarray([current.pos_x, current.pos_z], dtype=np.float32)
        end = np.asarray([next_waypoint.pos_x, next_waypoint.pos_z], dtype=np.float32)
        delta = end - start
        length_sq = float(np.dot(delta, delta))
        alpha = 0.0 if length_sq <= 1e-8 else _clip(float(np.dot(point - start, delta) / length_sq), 0.0, 1.0)
        projected_xz = start + alpha * delta
        distance = float(np.linalg.norm(point - projected_xz))
        segment_length = float(np.linalg.norm(delta))
        map_distance = float(current.distance_m + alpha * segment_length)

        candidate = {
            "distance": distance,
            "waypoint_index": waypoint_index,
            "next_waypoint_index": (waypoint_index + 1) % len(track_map.centerline),
            "alpha": alpha,
            "map_distance_m": map_distance,
            "projected_x": float(projected_xz[0]),
            "projected_y": _interpolate_scalar(current.pos_y, next_waypoint.pos_y, alpha),
            "projected_z": float(projected_xz[1]),
            "heading_x": _interpolate_scalar(current.heading_x, next_waypoint.heading_x, alpha),
            "heading_z": _interpolate_scalar(current.heading_z, next_waypoint.heading_z, alpha),
            "yaw_deg": _interpolate_scalar(current.yaw_deg, next_waypoint.yaw_deg, alpha),
            "curvature_radpm": _interpolate_scalar(current.curvature_radpm, next_waypoint.curvature_radpm, alpha),
            "delta_heading_2m_deg": _interpolate_scalar(current.delta_heading_2m_deg, next_waypoint.delta_heading_2m_deg, alpha),
            "curvature_score": _interpolate_scalar(current.curvature_score, next_waypoint.curvature_score, alpha),
            "reference_speed_mps": _interpolate_scalar(current.reference_speed_mps, next_waypoint.reference_speed_mps, alpha),
        }

        if best is None or candidate["distance"] < best["distance"]:
            best = candidate

    assert best is not None
    heading = np.asarray([best["heading_x"], best["heading_z"]], dtype=np.float32)
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm <= 1e-6:
        heading = _heading_from_yaw_deg(best["yaw_deg"])
    else:
        heading = heading / heading_norm

    if track_map.is_closed:
        map_distance = float(best["map_distance_m"] % track_map.total_length_m)
    else:
        map_distance = _clip(best["map_distance_m"], 0.0, track_map.total_length_m)

    return ProjectedTrackState(
        map_id=track_map.map_id,
        track_name=track_map.track_name,
        map_distance_m=map_distance,
        waypoint_index=int(best["waypoint_index"]),
        next_waypoint_index=int(best["next_waypoint_index"]),
        interpolation_alpha=float(best["alpha"]),
        pos_x=float(best["projected_x"]),
        pos_y=float(best["projected_y"]),
        pos_z=float(best["projected_z"]),
        heading_x=float(heading[0]),
        heading_z=float(heading[1]),
        yaw_deg=float(best["yaw_deg"]),
        distance_to_centerline_m=float(best["distance"]),
        curvature_radpm=float(best["curvature_radpm"]),
        delta_heading_2m_deg=float(best["delta_heading_2m_deg"]),
        curvature_score=float(best["curvature_score"]),
        reference_speed_mps=float(best["reference_speed_mps"]),
    )


def materialize_label_from_state(
    track_map: TrackMapArtifact,
    label_mode: str,
    map_distance_m: float,
    current_position_xyz: Sequence[float],
    yaw_deg: float,
    speed_mps: float,
    curvature_score: float,
    allow_tail_clamp: bool = False,
) -> Optional[StateLabel]:
    """Verili yol-durumundan tek bir (target_x, target_y) etiketi üretir.

    Mantık: bulunduğun mesafe + lookahead = ileri hedef mesafe -> merkez
    çizgisinde o noktanın dünya konumunu bul -> aracın ego-frame'ine çevir.
    lookahead, label_mode + hız + eğrilikten gelir. Açık pistte (kapalı
    değilse) yolun sonunu aşan hedef None döner (allow_tail_clamp ile sona
    sabitlenebilir). Bu, projenin ürettiği 'doğru cevabın' ta kendisidir."""
    requested_lookahead_m = lookahead_for_mode(
        label_mode=label_mode,
        speed_mps=float(speed_mps),
        curvature_score=float(curvature_score),
    )
    target_distance = float(map_distance_m) + float(requested_lookahead_m)
    tail_clamped = False
    if not track_map.is_closed and target_distance > track_map.total_length_m:
        if not allow_tail_clamp:
            return None
        tail_clamped = True
        target_distance = float(track_map.total_length_m)

    target_position = interpolate_centerline_position(
        track_map.centerline,
        total_length_m=track_map.total_length_m,
        distance_m=target_distance,
        is_closed=track_map.is_closed,
    )
    target_x, target_y = _world_to_ego_from_pose(
        current_position_xyz=current_position_xyz,
        yaw_deg=float(yaw_deg),
        target_position_xyz=target_position,
    )
    lookahead_m = float(target_distance - float(map_distance_m))
    if track_map.is_closed:
        lookahead_m = float(requested_lookahead_m)

    return StateLabel(
        label_mode=label_mode,
        lookahead_m=float(lookahead_m),
        target_x=float(target_x),
        target_y=float(target_y),
        tail_clamped=bool(tail_clamped),
    )


class LowNoiseTargetPointTeacher:
    """Faz 2/3 veri toplamada simülatörü süren 'ayrıcalıklı' teacher.

    'Ayrıcalıklı' çünkü simülatörün gerçek pose'unu görür (model göremez).
    Pisti takip eder ama kasıtlı bozmalar ekler: direksiyon/hedef gürültüsü,
    şerit sapması (lane bias) ve kurtarma (recovery) manevraları. Amaç:
    modelin gördüğü veriye çeşitlilik katmak, hatadan dönmeyi öğretmek.
    Her karede compute_action() ile hem komut hem TEMİZ etiketi üretir."""

    @classmethod
    def for_profile(
        cls,
        track_map: TrackMapArtifact,
        seed: int,
        control_hz: float,
        label_mode: str,
        collection_profile: str = COLLECTION_PROFILE_PHASE2,
        nominal_only: bool = False,
        fixed_throttle: float | None = None,
    ) -> "LowNoiseTargetPointTeacher":
        throttle_overrides = {}
        if fixed_throttle is not None:
            resolved_throttle = float(fixed_throttle)
            throttle_overrides = {
                "base_throttle": resolved_throttle,
                "min_throttle": resolved_throttle,
                "max_throttle": resolved_throttle,
                "steer_throttle_gain": 0.0,
                "curvature_throttle_gain": 0.0,
                "cte_throttle_gain": 0.0,
                "throttle_noise_sigma": 0.0,
                "throttle_noise_clip": 0.0,
                "throttle_scale_range": (0.0, 0.0),
                "recovery_throttle_multiplier": 1.0,
            }

        nominal_overrides = {}
        if nominal_only:
            nominal_overrides = {
                "target_noise_sigma_x": 0.0,
                "target_noise_sigma_y": 0.0,
                "steering_noise_sigma": 0.0,
                "throttle_noise_sigma": 0.0,
                "target_noise_clip_x": 0.0,
                "target_noise_clip_y": 0.0,
                "steering_noise_clip": 0.0,
                "throttle_noise_clip": 0.0,
                "lane_bias_range": (0.0, 0.0),
                "lane_bias_duration_steps": (0, 0),
                "lane_bias_gap_steps": (0, 0),
                "steering_lag_frames": 0,
            }
        profile_overrides = {**nominal_overrides, **throttle_overrides}

        if collection_profile == COLLECTION_PROFILE_PHASE2:
            return cls(
                track_map=track_map,
                seed=seed,
                control_hz=control_hz,
                label_mode=label_mode,
                collection_profile=collection_profile,
                nominal_only=nominal_only,
                **profile_overrides,
            )

        if collection_profile == COLLECTION_PROFILE_PHASE3:
            phase3_defaults = {
                "base_throttle": 0.22,
                "min_throttle": 0.09,
                "max_throttle": 0.22,
                "steer_throttle_gain": 0.10,
                "curvature_throttle_gain": 0.04,
                "cte_throttle_gain": 0.04,
                "target_noise_sigma_x": 0.05,
                "target_noise_sigma_y": 0.015,
                "steering_noise_sigma": 0.02,
                "throttle_noise_sigma": 0.012,
                "target_noise_clip_x": 0.12,
                "target_noise_clip_y": 0.05,
                "steering_noise_clip": 0.06,
                "throttle_noise_clip": 0.03,
                "noise_smoothing": 0.80,
                "recovery_warmup_steps": 24,
                "disturbance_gap_steps": (24, 40),
                "steering_bias_range": (0.15, 0.35),
                "target_offset_range": (0.35, 0.50),
                "throttle_scale_range": (0.15, 0.25),
                "disturbance_duration_steps": (6, 12),
                "recovery_settle_steps": 14,
                "lane_bias_range": (0.04, 0.10),
                "lane_bias_duration_steps": (20, 60),
                "lane_bias_gap_steps": (35, 80),
                "steering_lag_frames": 2,
                "disturbance_weights": ("steering_bias", "steering_bias", "target_offset", "throttle_scale"),
                "recovery_steer_gain_scale": 1.35,
                "recovery_noise_scale": 0.50,
                "recovery_throttle_multiplier": 0.90,
                "disable_lag_during_recovery": True,
                "disturbance_max_curvature_score": 0.65 if track_map.track_name == "donkey-warehouse-v0" else 0.75,
                "safety_override_cte_fraction": 0.90,
                "safety_override_centerline_m": 1.35,
                "safety_recenter_steps": max(6, int(round(float(control_hz) * 0.4))),
            }
            phase3_defaults.update(profile_overrides)
            return cls(
                track_map=track_map,
                seed=seed,
                control_hz=control_hz,
                label_mode=label_mode,
                collection_profile=collection_profile,
                nominal_only=nominal_only,
                **phase3_defaults,
            )

        raise ValueError(f"Unsupported collection profile: {collection_profile}")

    def __init__(
        self,
        track_map: TrackMapArtifact,
        seed: int = 42,
        control_hz: float = 20.0,
        label_mode: str = ADAPTIVE_V1,
        collection_profile: str = COLLECTION_PROFILE_PHASE2,
        nominal_only: bool = False,
        steer_gain: float = 1.35,
        base_throttle: float = 0.24,
        min_throttle: float = 0.10,
        max_throttle: float = 0.24,
        steer_throttle_gain: float = 0.08,
        curvature_throttle_gain: float = 0.03,
        cte_throttle_gain: float = 0.03,
        target_noise_sigma_x: float = 0.02,
        target_noise_sigma_y: float = 0.008,
        steering_noise_sigma: float = 0.01,
        throttle_noise_sigma: float = 0.01,
        target_noise_clip_x: float = 0.05,
        target_noise_clip_y: float = 0.02,
        steering_noise_clip: float = 0.03,
        throttle_noise_clip: float = 0.02,
        noise_smoothing: float = 0.70,
        recovery_warmup_steps: int = 30,
        disturbance_gap_steps: tuple[int, int] = (25, 50),
        steering_bias_range: tuple[float, float] = (0.10, 0.18),
        target_offset_range: tuple[float, float] = (0.15, 0.25),
        throttle_scale_range: tuple[float, float] = (0.10, 0.15),
        disturbance_duration_steps: tuple[int, int] = (4, 10),
        recovery_settle_steps: int = 12,
        lane_bias_range: tuple[float, float] = (0.0, 0.0),
        lane_bias_duration_steps: tuple[int, int] = (0, 0),
        lane_bias_gap_steps: tuple[int, int] = (0, 0),
        steering_lag_frames: int = 0,
        steer_smoothing: float = 0.65,
        max_steer_delta: float = 0.05,
        disturbance_weights: Sequence[str] = ("steering_bias", "target_offset", "throttle_scale"),
        recovery_steer_gain_scale: float = 1.0,
        recovery_noise_scale: float = 1.0,
        recovery_throttle_multiplier: float = 1.0,
        disable_lag_during_recovery: bool = False,
        disturbance_max_curvature_score: float = 1.0,
        safety_override_cte_fraction: float = 0.60,
        safety_override_centerline_m: float = 0.90,
        safety_recenter_steps: int = 20,
        max_safe_cte: float = 2.5,
    ) -> None:
        self.track_map = track_map
        self.seed = int(seed)
        self.rng = np.random.default_rng(int(seed))
        self.control_hz = float(control_hz)
        self.dt = 1.0 / max(self.control_hz, 1.0)
        self.label_mode = str(label_mode)
        self.collection_profile = str(collection_profile)
        self.nominal_only = bool(nominal_only)
        self.steer_gain = float(steer_gain)
        self.base_throttle = float(base_throttle)
        self.min_throttle = float(min_throttle)
        self.max_throttle = float(max_throttle)
        self.steer_throttle_gain = float(steer_throttle_gain)
        self.curvature_throttle_gain = float(curvature_throttle_gain)
        self.cte_throttle_gain = float(cte_throttle_gain)
        self.target_noise_sigma_x = float(target_noise_sigma_x)
        self.target_noise_sigma_y = float(target_noise_sigma_y)
        self.steering_noise_sigma = float(steering_noise_sigma)
        self.throttle_noise_sigma = float(throttle_noise_sigma)
        self.target_noise_clip_x = float(target_noise_clip_x)
        self.target_noise_clip_y = float(target_noise_clip_y)
        self.steering_noise_clip = float(steering_noise_clip)
        self.throttle_noise_clip = float(throttle_noise_clip)
        self.noise_smoothing = float(noise_smoothing)
        self.recovery_warmup_steps = int(recovery_warmup_steps)
        self.disturbance_gap_steps = (int(disturbance_gap_steps[0]), int(disturbance_gap_steps[1]))
        self.steering_bias_range = (float(steering_bias_range[0]), float(steering_bias_range[1]))
        self.target_offset_range = (float(target_offset_range[0]), float(target_offset_range[1]))
        self.throttle_scale_range = (float(throttle_scale_range[0]), float(throttle_scale_range[1]))
        self.disturbance_duration_steps = (int(disturbance_duration_steps[0]), int(disturbance_duration_steps[1]))
        self.recovery_settle_steps = int(recovery_settle_steps)
        self.lane_bias_range = (float(lane_bias_range[0]), float(lane_bias_range[1]))
        self.lane_bias_duration_steps = (int(lane_bias_duration_steps[0]), int(lane_bias_duration_steps[1]))
        self.lane_bias_gap_steps = (int(lane_bias_gap_steps[0]), int(lane_bias_gap_steps[1]))
        self.steering_lag_frames = int(steering_lag_frames)
        self.steer_smoothing = float(steer_smoothing)
        self.max_steer_delta = float(max_steer_delta)
        self.disturbance_weights = tuple(str(weight) for weight in disturbance_weights)
        self.recovery_steer_gain_scale = float(recovery_steer_gain_scale)
        self.recovery_noise_scale = float(recovery_noise_scale)
        self.recovery_throttle_multiplier = float(recovery_throttle_multiplier)
        self.disable_lag_during_recovery = bool(disable_lag_during_recovery)
        self.disturbance_max_curvature_score = float(disturbance_max_curvature_score)
        self.safety_override_cte_fraction = float(safety_override_cte_fraction)
        self.safety_override_centerline_m = float(safety_override_centerline_m)
        self.safety_recenter_steps = int(safety_recenter_steps)
        self.max_safe_cte = float(max_safe_cte)
        self._target_noise_x = 0.0
        self._target_noise_y = 0.0
        self._steering_noise = 0.0
        self._throttle_noise = 0.0
        self._lane_bias_offset_x = 0.0
        self._lane_bias_remaining = 0
        self._active_disturbance_type = "none"
        self._disturbance_remaining = 0
        self._recovery_steering_bias = 0.0
        self._recovery_target_offset_x = 0.0
        self._recovery_throttle_scale = 1.0
        self._next_disturbance_step = self.recovery_warmup_steps
        self._next_lane_bias_step = self.recovery_warmup_steps
        self._settle_remaining = 0
        self._safety_override_remaining = 0
        self._prev_commanded_steer: Optional[float] = None
        self._steering_lag_queue = deque([0.0] * self.steering_lag_frames, maxlen=self.steering_lag_frames + 1)

    def reset(self) -> None:
        self._target_noise_x = 0.0
        self._target_noise_y = 0.0
        self._steering_noise = 0.0
        self._throttle_noise = 0.0
        self._lane_bias_offset_x = 0.0
        self._lane_bias_remaining = 0
        self._active_disturbance_type = "none"
        self._disturbance_remaining = 0
        self._recovery_steering_bias = 0.0
        self._recovery_target_offset_x = 0.0
        self._recovery_throttle_scale = 1.0
        self._next_lane_bias_step = self.recovery_warmup_steps
        self._settle_remaining = 0
        self._safety_override_remaining = 0
        self._prev_commanded_steer = None
        self._steering_lag_queue = deque([0.0] * self.steering_lag_frames, maxlen=self.steering_lag_frames + 1)
        self._schedule_next_disturbance(self.recovery_warmup_steps)

    def summary(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "label_mode": self.label_mode,
            "collection_profile": self.collection_profile,
            "nominal_only": bool(self.nominal_only),
            "steer_gain": self.steer_gain,
            "base_throttle": self.base_throttle,
            "min_throttle": self.min_throttle,
            "max_throttle": self.max_throttle,
            "steer_throttle_gain": self.steer_throttle_gain,
            "curvature_throttle_gain": self.curvature_throttle_gain,
            "cte_throttle_gain": self.cte_throttle_gain,
            "target_noise_sigma_x": self.target_noise_sigma_x,
            "target_noise_sigma_y": self.target_noise_sigma_y,
            "steering_noise_sigma": self.steering_noise_sigma,
            "throttle_noise_sigma": self.throttle_noise_sigma,
            "target_noise_clip_x": self.target_noise_clip_x,
            "target_noise_clip_y": self.target_noise_clip_y,
            "steering_noise_clip": self.steering_noise_clip,
            "throttle_noise_clip": self.throttle_noise_clip,
            "noise_smoothing": self.noise_smoothing,
            "recovery_warmup_steps": self.recovery_warmup_steps,
            "disturbance_gap_min_steps": self.disturbance_gap_steps[0],
            "disturbance_gap_max_steps": self.disturbance_gap_steps[1],
            "disturbance_duration_min_steps": self.disturbance_duration_steps[0],
            "disturbance_duration_max_steps": self.disturbance_duration_steps[1],
            "recovery_settle_steps": self.recovery_settle_steps,
            "lane_bias_min": self.lane_bias_range[0],
            "lane_bias_max": self.lane_bias_range[1],
            "lane_bias_duration_min_steps": self.lane_bias_duration_steps[0],
            "lane_bias_duration_max_steps": self.lane_bias_duration_steps[1],
            "lane_bias_gap_min_steps": self.lane_bias_gap_steps[0],
            "lane_bias_gap_max_steps": self.lane_bias_gap_steps[1],
            "steering_lag_frames": self.steering_lag_frames,
            "steer_smoothing": self.steer_smoothing,
            "max_steer_delta": self.max_steer_delta,
            "recovery_steer_gain_scale": self.recovery_steer_gain_scale,
            "recovery_noise_scale": self.recovery_noise_scale,
            "recovery_throttle_multiplier": self.recovery_throttle_multiplier,
            "disable_lag_during_recovery": self.disable_lag_during_recovery,
            "disturbance_max_curvature_score": self.disturbance_max_curvature_score,
            "safety_override_cte_fraction": self.safety_override_cte_fraction,
            "safety_override_centerline_m": self.safety_override_centerline_m,
            "safety_recenter_steps": self.safety_recenter_steps,
            "steering_bias_min": self.steering_bias_range[0],
            "steering_bias_max": self.steering_bias_range[1],
            "target_offset_min": self.target_offset_range[0],
            "target_offset_max": self.target_offset_range[1],
            "throttle_scale_min": self.throttle_scale_range[0],
            "throttle_scale_max": self.throttle_scale_range[1],
            "max_safe_cte": self.max_safe_cte,
        }

    def _sample_signed_uniform(self, low: float, high: float) -> float:
        magnitude = float(self.rng.uniform(float(low), float(high)))
        return magnitude if bool(self.rng.integers(0, 2)) else -magnitude

    def _schedule_next_disturbance(self, current_step: int) -> None:
        gap = int(self.rng.integers(self.disturbance_gap_steps[0], self.disturbance_gap_steps[1] + 1))
        self._next_disturbance_step = int(current_step) + gap

    def _schedule_next_lane_bias(self, current_step: int) -> None:
        if self.lane_bias_gap_steps[1] <= 0:
            self._next_lane_bias_step = int(current_step) + 1_000_000
            return
        gap = int(self.rng.integers(self.lane_bias_gap_steps[0], self.lane_bias_gap_steps[1] + 1))
        self._next_lane_bias_step = int(current_step) + gap

    def _activate_disturbance(self, current_step: int) -> None:
        disturbance_type = str(self.rng.choice(self.disturbance_weights))
        duration = int(self.rng.integers(self.disturbance_duration_steps[0], self.disturbance_duration_steps[1] + 1))
        self._active_disturbance_type = disturbance_type
        self._disturbance_remaining = duration
        self._recovery_steering_bias = 0.0
        self._recovery_target_offset_x = 0.0
        self._recovery_throttle_scale = 1.0

        if disturbance_type == "steering_bias":
            self._recovery_steering_bias = self._sample_signed_uniform(*self.steering_bias_range)
        elif disturbance_type == "target_offset":
            self._recovery_target_offset_x = self._sample_signed_uniform(*self.target_offset_range)
        elif disturbance_type == "throttle_scale":
            scale_delta = self._sample_signed_uniform(*self.throttle_scale_range)
            self._recovery_throttle_scale = float(max(0.70, 1.0 + scale_delta))

        self._schedule_next_disturbance(current_step + duration)

    def _update_lane_bias(self, observation: SimObservation, projected_state: ProjectedTrackState, step_index: int, safety_override_active: bool) -> float:
        if self.lane_bias_range[1] <= 0.0:
            return 0.0

        if safety_override_active or observation.hit != "none" or observation.done:
            self._lane_bias_offset_x = 0.0
            self._lane_bias_remaining = 0
            self._schedule_next_lane_bias(step_index)
            return 0.0

        if self._lane_bias_remaining > 0:
            self._lane_bias_remaining -= 1
            if self._lane_bias_remaining == 0:
                self._lane_bias_offset_x = 0.0
                self._schedule_next_lane_bias(step_index)
            return float(self._lane_bias_offset_x)

        can_start = (
            step_index >= self._next_lane_bias_step
            and abs(float(observation.cte)) <= 0.45
            and float(projected_state.distance_to_centerline_m) <= 0.55
        )
        if not can_start:
            return float(self._lane_bias_offset_x)

        self._lane_bias_offset_x = self._sample_signed_uniform(*self.lane_bias_range)
        self._lane_bias_remaining = int(
            self.rng.integers(self.lane_bias_duration_steps[0], self.lane_bias_duration_steps[1] + 1)
        )
        return float(self._lane_bias_offset_x)

    def _update_safety_override(self, observation: SimObservation, projected_state: ProjectedTrackState) -> bool:
        risk_cte = abs(float(observation.cte)) > (self.safety_override_cte_fraction * self.max_safe_cte)
        risk_centerline = float(projected_state.distance_to_centerline_m) > self.safety_override_centerline_m
        risk_state = bool(risk_cte or risk_centerline or observation.hit != "none" or observation.done)

        if risk_state:
            self._safety_override_remaining = self.safety_recenter_steps
        elif self._safety_override_remaining > 0:
            centered = abs(float(observation.cte)) <= 0.30 and float(projected_state.distance_to_centerline_m) <= 0.35
            decay = 2 if centered else 1
            self._safety_override_remaining = max(0, self._safety_override_remaining - decay)

        return bool(risk_state or self._safety_override_remaining > 0)

    def _update_disturbance(self, observation: SimObservation, projected_state: ProjectedTrackState, step_index: int) -> tuple[str, str]:
        if self._disturbance_remaining > 0:
            self._disturbance_remaining -= 1
            if self._disturbance_remaining == 0:
                self._active_disturbance_type = "none"
                self._recovery_steering_bias = 0.0
                self._recovery_target_offset_x = 0.0
                self._recovery_throttle_scale = 1.0
                self._settle_remaining = self.recovery_settle_steps
            return "recovery", self._active_disturbance_type if self._active_disturbance_type != "none" else "settle"

        if self._settle_remaining > 0:
            if abs(float(observation.cte)) <= 0.30 and float(projected_state.distance_to_centerline_m) <= 0.35:
                self._settle_remaining = 0
            else:
                self._settle_remaining -= 1
                return "recovery", "settle"

        can_disturb = (
            not self.nominal_only
            and step_index >= self.recovery_warmup_steps
            and step_index >= self._next_disturbance_step
            and abs(float(observation.cte)) <= 0.55
            and float(projected_state.distance_to_centerline_m) <= 0.65
            and float(projected_state.curvature_score) <= self.disturbance_max_curvature_score
            and observation.hit == "none"
            and not observation.done
        )
        if can_disturb:
            self._activate_disturbance(step_index)
            self._disturbance_remaining -= 1
            return "recovery", self._active_disturbance_type
        return "nominal", "none"

    def _smoothed_noise(self, previous: float, sigma: float, clip_limit: float) -> float:
        noise = float(self.rng.normal(0.0, sigma))
        value = self.noise_smoothing * float(previous) + (1.0 - self.noise_smoothing) * noise
        return _clip(value, -clip_limit, clip_limit)

    def compute_action(self, observation: SimObservation, step_index: int) -> LowNoiseTeacherOutput:
        projected_state = project_pose_to_centerline(
            self.track_map,
            pos_x=float(observation.pos[0]),
            pos_y=float(observation.pos[1]),
            pos_z=float(observation.pos[2]),
        )
        speed_mps = max(float(observation.forward_vel), float(observation.speed), 0.0)
        clean_label = materialize_label_from_state(
            track_map=self.track_map,
            label_mode=self.label_mode,
            map_distance_m=projected_state.map_distance_m,
            current_position_xyz=observation.pos,
            yaw_deg=float(observation.yaw_deg),
            speed_mps=float(speed_mps),
            curvature_score=float(projected_state.curvature_score),
            allow_tail_clamp=True,
        )
        if clean_label is None:
            raise RuntimeError("Low-noise teacher could not materialize a control target from the current simulator state.")

        safety_override_active = self._update_safety_override(observation, projected_state)

        if safety_override_active:
            self._target_noise_x = 0.0
            self._target_noise_y = 0.0
            self._steering_noise = 0.0
            self._throttle_noise = 0.0
            self._lane_bias_offset_x = 0.0
            self._lane_bias_remaining = 0
            self._active_disturbance_type = "none"
            self._disturbance_remaining = 0
            self._recovery_steering_bias = 0.0
            self._recovery_target_offset_x = 0.0
            self._recovery_throttle_scale = 1.0
            self._settle_remaining = 0
            self._steering_lag_queue = deque([0.0] * self.steering_lag_frames, maxlen=self.steering_lag_frames + 1)
            if self.nominal_only:
                scenario, disturbance_type = ("nominal", "none")
            else:
                scenario, disturbance_type = ("recovery", "safety_override")
        else:
            scenario, disturbance_type = self._update_disturbance(observation, projected_state, step_index)
        recovery_control_active = bool(scenario == "recovery" and disturbance_type != "safety_override")

        if safety_override_active:
            lane_bias_offset_x = 0.0
        elif self.nominal_only:
            self._target_noise_x = 0.0
            self._target_noise_y = 0.0
            self._steering_noise = 0.0
            self._throttle_noise = 0.0
            self._lane_bias_offset_x = 0.0
            self._lane_bias_remaining = 0
            lane_bias_offset_x = 0.0
        else:
            self._target_noise_x = self._smoothed_noise(self._target_noise_x, self.target_noise_sigma_x, self.target_noise_clip_x)
            self._target_noise_y = self._smoothed_noise(self._target_noise_y, self.target_noise_sigma_y, self.target_noise_clip_y)
            self._steering_noise = self._smoothed_noise(self._steering_noise, self.steering_noise_sigma, self.steering_noise_clip)
            self._throttle_noise = self._smoothed_noise(self._throttle_noise, self.throttle_noise_sigma, self.throttle_noise_clip)
            lane_bias_offset_x = self._update_lane_bias(observation, projected_state, step_index, safety_override_active)

        if recovery_control_active:
            self._lane_bias_offset_x = 0.0
            self._lane_bias_remaining = 0
            lane_bias_offset_x = 0.0

        steering_noise = float(self._steering_noise)
        throttle_noise = float(self._throttle_noise)
        if recovery_control_active:
            steering_noise *= self.recovery_noise_scale
            throttle_noise *= self.recovery_noise_scale

        total_noise_target_x = float(self._target_noise_x + lane_bias_offset_x)
        applied_target_x = float(clean_label.target_x + total_noise_target_x + self._recovery_target_offset_x)
        applied_target_y = float(max(0.20, clean_label.target_y + self._target_noise_y))
        heading_error = math.atan2(applied_target_x, max(applied_target_y, 1e-3))

        steer_gain = self.steer_gain * (self.recovery_steer_gain_scale if recovery_control_active else 1.0)
        steer_cmd = steer_gain * heading_error
        steer_cmd += steering_noise
        steer_cmd += self._recovery_steering_bias
        if observation.hit != "none" or observation.done:
            steer_cmd = 0.0
        steer_cmd = _clip(steer_cmd, -1.0, 1.0)
        steer_cmd = _blend_with_rate_limit(
            self._prev_commanded_steer,
            steer_cmd,
            smoothing=self.steer_smoothing,
            max_delta=self.max_steer_delta,
        )
        steer_cmd = _clip(steer_cmd, -1.0, 1.0)
        commanded_steer = float(steer_cmd)
        self._prev_commanded_steer = commanded_steer

        if self.steering_lag_frames > 0 and not safety_override_active and not (recovery_control_active and self.disable_lag_during_recovery):
            self._steering_lag_queue.append(commanded_steer)
            steer_cmd = float(self._steering_lag_queue.popleft())
        else:
            if self.steering_lag_frames > 0:
                self._steering_lag_queue = deque([commanded_steer] * self.steering_lag_frames, maxlen=self.steering_lag_frames + 1)
            steer_cmd = commanded_steer

        throttle_cmd = self.base_throttle
        throttle_cmd -= self.steer_throttle_gain * abs(float(steer_cmd))
        throttle_cmd -= self.curvature_throttle_gain * abs(float(projected_state.curvature_score))
        throttle_cmd -= self.cte_throttle_gain * min(abs(float(observation.cte)), 1.0)
        throttle_cmd *= self._recovery_throttle_scale
        if recovery_control_active:
            throttle_cmd *= self.recovery_throttle_multiplier
        throttle_cmd += throttle_noise
        if abs(float(observation.cte)) > self.max_safe_cte:
            throttle_cmd = self.min_throttle
        throttle_cmd = _clip(throttle_cmd, self.min_throttle, self.max_throttle)
        commanded_throttle = float(throttle_cmd)

        brake_cmd = 0.0
        if observation.hit != "none":
            throttle_cmd = 0.0
            brake_cmd = 1.0

        action = TeacherAction(
            steering=float(steer_cmd),
            throttle=float(throttle_cmd),
            brake=float(brake_cmd),
        )
        return LowNoiseTeacherOutput(
            action=action,
            projected_state=projected_state,
            clean_label=clean_label,
            collection_profile=self.collection_profile,
            noise_target_x=float(self._target_noise_x),
            noise_target_y=float(self._target_noise_y),
            lane_bias_offset_x=float(lane_bias_offset_x),
            steering_noise=float(steering_noise),
            throttle_noise=float(throttle_noise),
            teacher_commanded_steering=float(commanded_steer),
            teacher_commanded_throttle=float(commanded_throttle),
            steering_lag_frames=int(self.steering_lag_frames),
            safety_override_active=bool(safety_override_active),
            recovery_steering_bias=float(self._recovery_steering_bias),
            recovery_target_offset_x=float(self._recovery_target_offset_x),
            recovery_throttle_scale=float(self._recovery_throttle_scale),
            applied_target_x=float(applied_target_x),
            applied_target_y=float(applied_target_y),
            applied_lookahead_m=float(clean_label.lookahead_m),
            scenario=str(scenario),
            disturbance_type=str(disturbance_type),
        )


class CleanMappingTeacher:
    """Faz 1 harita çıkarımı için güvenlik öncelikli, temiz süren teacher.

    Simülatörün gerçek durumunu (cte=merkeze uzaklık, yaw=yön) kullanıp
    pisti olabildiğince ortadan ve sakin sürer. Gürültü EKLEMEZ; amacı
    pürüzsüz bir referans iz (centerline) çıkarmaktır. Bu iz daha sonra
    tüm etiketlerin hesaplandığı 'pist haritası' olur."""

    def __init__(
        self,
        control_hz: float = 20.0,
        probe_steer: float = 0.20,
        probe_throttle: float = 0.16,
        probe_steps: int = 12,
        cte_gain: float = 1.20,
        dcte_gain: float = 0.35,
        yaw_rate_gain: float = 0.05,
        base_throttle: float = 0.20,
        min_throttle: float = 0.08,
        throttle_cte_gain: float = 0.05,
        throttle_steer_gain: float = 0.12,
        steer_smoothing: float = 0.75,
        max_steer_delta: float = 0.05,
        max_steer: float = 1.00,
        max_safe_cte: float = 3.0,
    ) -> None:
        self.control_hz = float(control_hz)
        self.dt = 1.0 / max(self.control_hz, 1.0)
        self.probe_steer = float(probe_steer)
        self.probe_throttle = float(probe_throttle)
        self.probe_steps = int(probe_steps)
        self.cte_gain = float(cte_gain)
        self.dcte_gain = float(dcte_gain)
        self.yaw_rate_gain = float(yaw_rate_gain)
        self.base_throttle = float(base_throttle)
        self.min_throttle = float(min_throttle)
        self.throttle_cte_gain = float(throttle_cte_gain)
        self.throttle_steer_gain = float(throttle_steer_gain)
        self.steer_smoothing = float(steer_smoothing)
        self.max_steer_delta = float(max_steer_delta)
        self.max_steer = float(max_steer)
        self.max_safe_cte = float(max_safe_cte)
        self.cte_correction_sign = -1.0
        self._prev_cte = None
        self._prev_yaw = None
        self._prev_steer = None

    def reset(self) -> None:
        self._prev_cte = None
        self._prev_yaw = None
        self._prev_steer = None

    def summary(self) -> Dict[str, float]:
        return {
            "cte_gain": self.cte_gain,
            "dcte_gain": self.dcte_gain,
            "yaw_rate_gain": self.yaw_rate_gain,
            "base_throttle": self.base_throttle,
            "min_throttle": self.min_throttle,
            "throttle_cte_gain": self.throttle_cte_gain,
            "throttle_steer_gain": self.throttle_steer_gain,
            "steer_smoothing": self.steer_smoothing,
            "max_steer_delta": self.max_steer_delta,
            "max_steer": self.max_steer,
            "max_safe_cte": self.max_safe_cte,
            "probe_steer": self.probe_steer,
            "probe_throttle": self.probe_throttle,
            "probe_steps": self.probe_steps,
            "cte_correction_sign": self.cte_correction_sign,
        }

    def infer_cte_correction_sign(self, session: DonkeySimSession) -> float:
        self.reset()
        measurements: List[float] = []
        session.reset()
        for _ in range(self.probe_steps):
            observation = session.step(self.probe_steer, self.probe_throttle, 0.0)
            measurements.append(float(observation.cte))
            if observation.done or observation.hit != "none":
                break

        session.reset()
        self.reset()

        if len(measurements) < 4:
            self.cte_correction_sign = -1.0
            return self.cte_correction_sign

        delta_cte = float(np.median(measurements[-3:]) - measurements[0])
        if abs(delta_cte) < 0.02:
            self.cte_correction_sign = -1.0
        else:
            self.cte_correction_sign = -1.0 if delta_cte > 0.0 else 1.0
        return self.cte_correction_sign

    def compute_action(self, observation: SimObservation) -> TeacherAction:
        if observation.done or observation.hit != "none" or abs(observation.cte) > self.max_safe_cte:
            return TeacherAction(steering=0.0, throttle=0.0, brake=1.0)

        cte_rate = 0.0
        if self._prev_cte is not None:
            cte_rate = (float(observation.cte) - float(self._prev_cte)) / self.dt

        yaw_rate = 0.0
        if self._prev_yaw is not None:
            yaw_rate = _wrap_delta_deg(float(observation.yaw_deg), float(self._prev_yaw)) / self.dt

        steer_cmd = self.cte_correction_sign * (
            self.cte_gain * float(observation.cte)
            + self.dcte_gain * float(cte_rate)
            + self.yaw_rate_gain * float(yaw_rate)
        )
        steer_cmd = _clip(steer_cmd, -self.max_steer, self.max_steer)
        steer_cmd = _blend_with_rate_limit(
            self._prev_steer,
            steer_cmd,
            smoothing=self.steer_smoothing,
            max_delta=self.max_steer_delta,
        )
        steer_cmd = _clip(steer_cmd, -self.max_steer, self.max_steer)

        throttle_cmd = self.base_throttle
        throttle_cmd -= self.throttle_cte_gain * abs(float(observation.cte))
        throttle_cmd -= self.throttle_steer_gain * abs(float(steer_cmd))
        throttle_cmd = _clip(throttle_cmd, self.min_throttle, self.base_throttle)

        self._prev_cte = float(observation.cte)
        self._prev_yaw = float(observation.yaw_deg)
        self._prev_steer = float(steer_cmd)

        return TeacherAction(steering=float(steer_cmd), throttle=float(throttle_cmd), brake=0.0)


def fixed_1p2m_lookahead(speed_mps: float, curvature_score: float) -> float:
    """Sabit lookahead modu: ne olursa olsun 1.2 m ileriye bak.
    Basit; hız/viraj dikkate alınmaz (parametreler yok sayılır)."""
    del speed_mps, curvature_score
    return 1.2


def adaptive_v1_lookahead(speed_mps: float, curvature_score: float) -> float:
    """Uyarlanır lookahead: hız arttıkça ileriye, viraj keskinleştikçe yakına bak.
    1.0 + 0.30*hız - 0.75*eğrilik, [0.9, 2.2] m arasına sıkıştırılır. Düz/hızlıda
    erken dönüş için uzağa, keskin virajda hassasiyet için yakına bakar."""
    return _clip(1.0 + 0.30 * float(speed_mps) - 0.75 * float(curvature_score), 0.9, 2.2)


def lookahead_for_mode(label_mode: str, speed_mps: float, curvature_score: float) -> float:
    """Seçili etiket moduna göre doğru lookahead fonksiyonunu çağırır.
    Desteklenen: 'fixed_1p2m' ve 'adaptive_v1'; başka mod hata verir."""
    if label_mode == FIXED_1P2M:
        return fixed_1p2m_lookahead(speed_mps, curvature_score)
    if label_mode == ADAPTIVE_V1:
        return adaptive_v1_lookahead(speed_mps, curvature_score)
    raise ValueError(f"Unsupported label mode: {label_mode}")


def _world_to_ego(waypoint: CenterlineWaypoint, target_position: np.ndarray) -> tuple[float, float]:
    """Dünya koordinatındaki hedef noktayı bir waypoint'in ego-frame'ine çevirir.
    Waypoint'in yön vektörünü 'ileri' (y), sağ dikini 'sağ' (x) alır; hedefe
    olan farkı bu eksenlere izdüşürür. Döner: (target_x, target_y) metre."""
    current = np.asarray([waypoint.pos_x, waypoint.pos_z], dtype=np.float32)
    heading = np.asarray([waypoint.heading_x, waypoint.heading_z], dtype=np.float32)
    right = np.asarray([heading[1], -heading[0]], dtype=np.float32)
    delta = target_position[[0, 2]] - current
    target_x = float(np.dot(delta, right))
    target_y = float(np.dot(delta, heading))
    return target_x, target_y


def materialize_label_mode(track_map: TrackMapArtifact, label_mode: str) -> List[LabelRecord]:
    """Bir pist haritasının HER merkez-çizgi noktası için etiket üretir (Faz 1).

    Her waypoint'te: o noktanın hız/eğriliğine göre lookahead hesapla -> o kadar
    ileri noktayı bul -> ego-frame'e çevir -> LabelRecord olarak biriktir.
    Açık pistte yolun sonunu aşan noktalar atlanır. Çıktı, build_target_point_labels.py
    tarafından labels_<mode>.csv olarak yazılan 'ideal etiket' tablosudur."""
    records: List[LabelRecord] = []
    for waypoint in track_map.centerline:
        lookahead_m = lookahead_for_mode(
            label_mode=label_mode,
            speed_mps=waypoint.reference_speed_mps,
            curvature_score=waypoint.curvature_score,
        )
        if not track_map.is_closed and (waypoint.distance_m + lookahead_m) > track_map.total_length_m:
            continue
        target_position = interpolate_centerline_position(
            track_map.centerline,
            total_length_m=track_map.total_length_m,
            distance_m=waypoint.distance_m + lookahead_m,
            is_closed=track_map.is_closed,
        )
        target_x, target_y = _world_to_ego(waypoint, target_position)
        records.append(
            LabelRecord(
                map_id=track_map.map_id,
                track_name=track_map.track_name,
                label_mode=label_mode,
                waypoint_index=waypoint.waypoint_index,
                distance_m=waypoint.distance_m,
                reference_speed_mps=waypoint.reference_speed_mps,
                curvature_radpm=waypoint.curvature_radpm,
                delta_heading_2m_deg=waypoint.delta_heading_2m_deg,
                curvature_score=waypoint.curvature_score,
                lookahead_m=float(lookahead_m),
                target_x=float(target_x),
                target_y=float(target_y),
            )
        )
    return records


def materialize_dual_label_modes(track_map: TrackMapArtifact) -> Dict[str, List[LabelRecord]]:
    return {
        FIXED_1P2M: materialize_label_mode(track_map, FIXED_1P2M),
        ADAPTIVE_V1: materialize_label_mode(track_map, ADAPTIVE_V1),
    }


def _bucket_label(value: float, bins: Sequence[float]) -> str:
    pairs = list(zip(bins[:-1], bins[1:]))
    for index, (lower, upper) in enumerate(pairs):
        is_last = index == len(pairs) - 1
        upper_label = "+inf" if math.isinf(upper) else f"{upper:.2f}"
        if math.isinf(upper):
            if lower <= float(value):
                return f"[{lower:.2f},{upper_label})"
            continue
        if is_last:
            if lower <= float(value) <= upper:
                return f"[{lower:.2f},{upper_label}]"
        elif lower <= float(value) < upper:
            return f"[{lower:.2f},{upper_label})"
    final_lower = bins[-2]
    final_upper = bins[-1]
    final_upper_label = "+inf" if math.isinf(final_upper) else f"{final_upper:.2f}"
    return f"[{final_lower:.2f},{final_upper_label}]"


def _numeric_summary(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float32)
    if array.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None, "p5": None, "p50": None, "p95": None}

    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p5": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def summarize_lookahead_stats(track_map: TrackMapArtifact, label_mode: str, records: Sequence[LabelRecord]) -> Dict[str, object]:
    by_speed_bin: Dict[str, List[float]] = {}
    by_curvature_bin: Dict[str, List[float]] = {}

    for record in records:
        by_speed_bin.setdefault(_bucket_label(record.reference_speed_mps, LOOKAHEAD_SPEED_BINS), []).append(record.lookahead_m)
        by_curvature_bin.setdefault(_bucket_label(record.curvature_score, LOOKAHEAD_CURVATURE_BINS), []).append(record.lookahead_m)

    return {
        "map_id": track_map.map_id,
        "track_name": track_map.track_name,
        "label_mode": label_mode,
        "overall": _numeric_summary(record.lookahead_m for record in records),
        "by_speed_bin": {label: _numeric_summary(values) for label, values in sorted(by_speed_bin.items())},
        "by_curvature_bin": {label: _numeric_summary(values) for label, values in sorted(by_curvature_bin.items())},
    }


def label_records_to_rows(records: Sequence[LabelRecord]) -> List[Dict[str, object]]:
    return [asdict(record) for record in records]
