"""Bölüm bazında görüntü-domaini rastgeleleştirme (domain randomization).

Sim-to-real için kilit teknik. Simülasyon görüntüsü hep aynı 'temiz' görünür;
gerçek dünya ise değişken (ışık, yol dokusu, renk, gölge). Domain randomization,
her sürüş bölümüne farklı görsel 'tema' (yol/kenar/çevre görünümü, parlaklık,
kontrast, doygunluk, RGB kayması...) uygulayarak modeli TEK bir görünüme
bağlı kalmaktan kurtarır. Böylece gerçek kameraya aktarımda daha dayanıklı olur
(özellikle model_02_sim_domain_randomization config'inde agresif kullanılır).

Deterministiktir: (seed, pist, split, bölüm indeksi) aynıysa aynı profili üretir,
yani sonuçlar tekrarlanabilir.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


ROAD_APPEARANCES = ("dark_asphalt", "light_asphalt", "faded_concrete", "dusty_tan", "cool_gray")
EDGE_APPEARANCES = ("bright_yellow", "pale_yellow", "white_edge", "low_contrast_edge", "shadowed_edge")
ENVIRONMENT_APPEARANCES = ("warehouse_indoor", "open_generated", "narrow_walled", "racetrack_green", "bright_overcast")


@dataclass(frozen=True)
class DomainProfile:
    """Bir bölüme uygulanacak görsel tema tanımı (renk/parlaklık/görünüm seçimleri)."""
    domain_profile_id: str
    road_appearance: str
    edge_appearance: str
    environment_appearance: str
    brightness: float
    contrast: float
    saturation: float
    rgb_shift_r: int
    rgb_shift_g: int
    rgb_shift_b: int
    blur_radius: float
    jpeg_quality: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def sample_domain_profile(seed: int, track_name: str, split: str, episode_index: int) -> DomainProfile:
    """Bir bölüm için rastgele ama deterministik görsel profil üretir.
    Anahtar (seed, pist, split, bölüm) hep aynı profili verir. Yol/kenar/çevre
    görünümü ve renk/parlaklık parametrelerini seçer. Döner: DomainProfile."""
    rng = random.Random(f"{seed}:{track_name}:{split}:{episode_index}")
    road = str(rng.choice(ROAD_APPEARANCES))
    edge = str(rng.choice(EDGE_APPEARANCES))
    environment = str(rng.choice(ENVIRONMENT_APPEARANCES))
    brightness = float(rng.uniform(0.92, 1.08))
    contrast = float(rng.uniform(0.94, 1.08))
    saturation = float(rng.uniform(0.92, 1.08))
    rgb_shift_r = int(rng.randint(-6, 6))
    rgb_shift_g = int(rng.randint(-6, 6))
    rgb_shift_b = int(rng.randint(-6, 6))
    blur_radius = float(rng.choice((0.0, 0.0, 0.5, 1.0)))
    jpeg_quality = int(rng.randint(88, 95))
    digest = hashlib.sha1(
        f"{seed}:{track_name}:{split}:{episode_index}:{road}:{edge}:{environment}".encode("utf-8")
    ).hexdigest()[:8]
    return DomainProfile(
        domain_profile_id=f"{split}_{track_name}_{digest}",
        road_appearance=road,
        edge_appearance=edge,
        environment_appearance=environment,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        rgb_shift_r=rgb_shift_r,
        rgb_shift_g=rgb_shift_g,
        rgb_shift_b=rgb_shift_b,
        blur_radius=blur_radius,
        jpeg_quality=jpeg_quality,
    )


def apply_domain_profile(image: np.ndarray, profile: DomainProfile) -> np.ndarray:
    """Verilen görsel profili tek bir görüntüye uygular (renk/parlaklık vb.).
    sample_domain_profile'ın seçtiği dönüşümleri gerçekleştirir. Döner: dönüştürülmüş görüntü."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(image, mode="RGB")
    pil_image = ImageEnhance.Brightness(pil_image).enhance(profile.brightness)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(profile.contrast)
    pil_image = ImageEnhance.Color(pil_image).enhance(profile.saturation)

    shifted = np.asarray(pil_image, dtype=np.int16)
    shifted[..., 0] += int(profile.rgb_shift_r)
    shifted[..., 1] += int(profile.rgb_shift_g)
    shifted[..., 2] += int(profile.rgb_shift_b)
    pil_image = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8), mode="RGB")

    if float(profile.blur_radius) > 0.0:
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=float(profile.blur_radius)))

    return np.asarray(pil_image, dtype=np.uint8)
