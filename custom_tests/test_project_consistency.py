import json
from types import SimpleNamespace

import numpy as np

from target_point.controller import target_point_to_controls
from target_point.model import preprocess_image


def test_dynamic_throttle_reduces_speed_on_large_heading_error():
    straight_steering, straight_throttle = target_point_to_controls(
        target_x=0.0,
        target_y=1.0,
        steer_gain=1.0,
        steer_sign=1.0,
        throttle=0.2,
        min_forward=0.05,
        dynamic_throttle=True,
        base_throttle=0.35,
        min_throttle=0.12,
        curvature_throttle_angle_deg=15.0,
    )
    turn_steering, turn_throttle = target_point_to_controls(
        target_x=0.5,
        target_y=1.0,
        steer_gain=1.0,
        steer_sign=1.0,
        throttle=0.2,
        min_forward=0.05,
        dynamic_throttle=True,
        base_throttle=0.35,
        min_throttle=0.12,
        curvature_throttle_angle_deg=15.0,
    )

    assert straight_steering == 0.0
    assert straight_throttle == 0.35
    assert turn_steering > 0.0
    assert turn_throttle == 0.12


def test_preprocess_image_respects_target_point_dimensions():
    cfg = SimpleNamespace(
        TARGET_POINT_IMAGE_W=64,
        TARGET_POINT_IMAGE_H=32,
        TARGET_POINT_CROP_TOP=10,
        TARGET_POINT_CROP_BOTTOM=10,
        TARGET_POINT_CROP_LEFT=5,
        TARGET_POINT_CROP_RIGHT=5,
    )
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :, 1] = 255

    processed = preprocess_image(image, cfg)

    assert processed.shape == (32, 64, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_export_manifest_references_existing_model_artifacts():
    manifest_path = "models/model_export_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest
    for entry in manifest:
        assert entry["status"] == "ok"
        assert entry["params"] > 0
        for key in ("source", "h5", "tflite_fp16"):
            path = "models/" + entry[key]
            with open(path, "rb") as artifact:
                assert artifact.read(1)


def test_real_track_prep_requires_acceptance_logging():
    with open("docs/REAL_TRACK_PREP.md", "r", encoding="utf-8") as f:
        content = f.read()

    assert "Real track acceptance log" in content
    assert "lap count" in content
    assert "manual takeover count" in content
