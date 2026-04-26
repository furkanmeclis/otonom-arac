"""Deterministic clip-consistent augmentations for target-point training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from target_point.dataset import TargetPointSample


def _cfg_float(cfg, key: str, default: float) -> float:
    return float(getattr(cfg, key, default))


def _cfg_int(cfg, key: str, default: int) -> int:
    return int(getattr(cfg, key, default))


@dataclass(frozen=True)
class TemporalAugmentationParams:
    brightness: float
    contrast: float
    blur_radius: float
    rgb_shift_r: int
    rgb_shift_g: int
    rgb_shift_b: int
    shift_x_px: float
    rotation_deg: float
    perspective_top_px: float
    perspective_bottom_px: float


class TemporalAugmenter:
    """Apply deterministic augmentations shared within a short temporal clip."""

    def __init__(self, cfg, enabled: bool, seed: int) -> None:
        self.enabled = bool(enabled)
        self.seed = int(seed)
        self.epoch_index = 0
        self.clip_frames = max(1, _cfg_int(cfg, "TARGET_POINT_AUG_CLIP_FRAMES", 5))
        self.brightness_limit = max(0.0, _cfg_float(cfg, "TARGET_POINT_AUG_BRIGHTNESS_LIMIT", 0.20))
        self.contrast_limit = max(0.0, _cfg_float(cfg, "TARGET_POINT_AUG_CONTRAST_LIMIT", 0.20))
        self.blur_radius_max = max(0.0, _cfg_float(cfg, "TARGET_POINT_AUG_BLUR_RADIUS_MAX", 1.0))
        self.rgb_shift_max = max(0, _cfg_int(cfg, "TARGET_POINT_AUG_RGB_SHIFT_MAX", 12))
        self.shift_px_max = max(0.0, _cfg_float(cfg, "TARGET_POINT_AUG_SHIFT_PX_MAX", 6.0))
        self.rotation_deg_max = max(0.0, _cfg_float(cfg, "TARGET_POINT_AUG_ROTATION_DEG_MAX", 2.5))
        self.perspective_px_max = max(0.0, _cfg_float(cfg, "TARGET_POINT_AUG_PERSPECTIVE_PX_MAX", 5.0))

    def set_epoch(self, epoch_index: int) -> None:
        self.epoch_index = int(epoch_index)

    def _clip_id_for_sample(self, sample: TargetPointSample) -> int:
        return int(sample.frame_index) // self.clip_frames

    def _params_for_sample(self, sample: TargetPointSample) -> TemporalAugmentationParams:
        clip_id = self._clip_id_for_sample(sample)
        key = f"{self.seed}|{self.epoch_index}|{sample.episode_id}|{clip_id}".encode("utf-8")
        digest = hashlib.sha1(key).digest()
        rng_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(rng_seed)
        return TemporalAugmentationParams(
            brightness=float(1.0 + rng.uniform(-self.brightness_limit, self.brightness_limit)),
            contrast=float(1.0 + rng.uniform(-self.contrast_limit, self.contrast_limit)),
            blur_radius=float(rng.uniform(0.0, self.blur_radius_max)),
            rgb_shift_r=int(rng.integers(-self.rgb_shift_max, self.rgb_shift_max + 1)),
            rgb_shift_g=int(rng.integers(-self.rgb_shift_max, self.rgb_shift_max + 1)),
            rgb_shift_b=int(rng.integers(-self.rgb_shift_max, self.rgb_shift_max + 1)),
            shift_x_px=float(rng.uniform(-self.shift_px_max, self.shift_px_max)),
            rotation_deg=float(rng.uniform(-self.rotation_deg_max, self.rotation_deg_max)),
            perspective_top_px=float(rng.uniform(-self.perspective_px_max, self.perspective_px_max)),
            perspective_bottom_px=float(rng.uniform(-self.perspective_px_max, self.perspective_px_max)),
        )

    def _apply_spatial_transforms(self, image: Image.Image, params: TemporalAugmentationParams) -> Image.Image:
        fill = (0, 0, 0)
        width, height = image.size

        if abs(params.shift_x_px) > 1e-6:
            image = image.transform(
                image.size,
                Image.AFFINE,
                (1.0, 0.0, float(params.shift_x_px), 0.0, 1.0, 0.0),
                resample=Image.BILINEAR,
                fillcolor=fill,
            )

        if abs(params.rotation_deg) > 1e-6:
            image = image.rotate(params.rotation_deg, resample=Image.BILINEAR, fillcolor=fill)

        if abs(params.perspective_top_px) > 1e-6 or abs(params.perspective_bottom_px) > 1e-6:
            quad = (
                float(params.perspective_top_px),
                0.0,
                float(width + params.perspective_top_px),
                0.0,
                float(width + params.perspective_bottom_px),
                float(height),
                float(params.perspective_bottom_px),
                float(height),
            )
            image = image.transform(
                image.size,
                Image.QUAD,
                quad,
                resample=Image.BILINEAR,
                fillcolor=fill,
            )
        return image

    def apply(self, image_array: np.ndarray, sample: TargetPointSample, return_params: bool = False):
        if not self.enabled:
            identity = TemporalAugmentationParams(
                brightness=1.0,
                contrast=1.0,
                blur_radius=0.0,
                rgb_shift_r=0,
                rgb_shift_g=0,
                rgb_shift_b=0,
                shift_x_px=0.0,
                rotation_deg=0.0,
                perspective_top_px=0.0,
                perspective_bottom_px=0.0,
            )
            image_array = np.asarray(image_array, dtype=np.uint8)
            if return_params:
                return image_array, identity
            return image_array

        params = self._params_for_sample(sample)
        image = Image.fromarray(np.asarray(image_array, dtype=np.uint8))
        image = self._apply_spatial_transforms(image, params)
        if abs(params.brightness - 1.0) > 1e-6:
            image = ImageEnhance.Brightness(image).enhance(params.brightness)
        if abs(params.contrast - 1.0) > 1e-6:
            image = ImageEnhance.Contrast(image).enhance(params.contrast)
        if params.blur_radius > 1e-6:
            image = image.filter(ImageFilter.GaussianBlur(radius=params.blur_radius))

        image_array = np.asarray(image, dtype=np.int16)
        image_array[..., 0] += int(params.rgb_shift_r)
        image_array[..., 1] += int(params.rgb_shift_g)
        image_array[..., 2] += int(params.rgb_shift_b)
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        if return_params:
            return image_array, params
        return image_array
