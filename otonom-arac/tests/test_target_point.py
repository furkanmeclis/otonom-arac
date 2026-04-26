import tempfile
from pathlib import Path

import numpy as np

from target_point.controller import target_point_to_controls
from target_point.dataset import compute_cumulative_distances, compute_target_point, resolve_tub_paths
from target_point.diagnostics import evaluate_collapse_gate


def test_compute_target_point_is_centered_on_straight_path():
    positions = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    cumulative = compute_cumulative_distances(positions)

    target_x, target_y = compute_target_point(positions, cumulative, index=1, lookahead_meters=0.5)

    assert abs(target_x) < 1e-6
    assert target_y > 0.0


def test_compute_target_point_keeps_right_side_positive():
    positions = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 3.0],
        ],
        dtype=np.float32,
    )
    cumulative = compute_cumulative_distances(positions)

    target_x, target_y = compute_target_point(positions, cumulative, index=1, lookahead_meters=1.5)
    steering, throttle = target_point_to_controls(
        target_x=target_x,
        target_y=target_y,
        steer_gain=1.0,
        steer_sign=1.0,
        throttle=0.2,
        min_forward=0.05,
    )

    assert target_x > 0.0
    assert steering > 0.0
    assert throttle == 0.2


def test_compute_target_point_keeps_left_side_negative():
    positions = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [-1.0, 3.0],
        ],
        dtype=np.float32,
    )
    cumulative = compute_cumulative_distances(positions)

    target_x, target_y = compute_target_point(positions, cumulative, index=1, lookahead_meters=1.5)
    steering, throttle = target_point_to_controls(
        target_x=target_x,
        target_y=target_y,
        steer_gain=1.0,
        steer_sign=1.0,
        throttle=0.2,
        min_forward=0.05,
    )

    assert target_x < 0.0
    assert steering < 0.0
    assert throttle == 0.2


def test_target_point_controller_stops_when_target_is_not_forward():
    steering, throttle = target_point_to_controls(
        target_x=0.1,
        target_y=0.01,
        steer_gain=1.0,
        steer_sign=1.0,
        throttle=0.2,
        min_forward=0.05,
    )

    assert steering == 0.0
    assert throttle == 0.0


def test_resolve_tub_paths_prefers_child_tubs_over_parent_root():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "manifest.json").write_text("[]", encoding="utf-8")
        child_a = root / "tub_a"
        child_b = root / "tub_b"
        child_a.mkdir()
        child_b.mkdir()
        (child_a / "manifest.json").write_text("[]", encoding="utf-8")
        (child_b / "manifest.json").write_text("[]", encoding="utf-8")

        resolved = resolve_tub_paths(str(root))

    assert resolved == [child_a.resolve().as_posix(), child_b.resolve().as_posix()]


def test_collapse_gate_detects_mean_regression():
    train_metrics = {
        "pred_x_std_ratio": 0.05,
        "corr_x": 0.1,
        "segment_metrics": {"turn": {"mae_x": 0.5}},
    }
    val_metrics = {
        "pred_x_std_ratio": 0.03,
        "corr_x": 0.0,
        "segment_metrics": {"turn": {"mae_x": 0.4}},
    }

    gate = evaluate_collapse_gate(train_metrics, val_metrics)

    assert gate["passed"] is False
    assert gate["checks"]["val_pred_x_std_ratio"] is False
    assert gate["checks"]["train_pred_x_std_ratio"] is False
    assert gate["checks"]["val_corr_x"] is False
    assert gate["checks"]["turn_mae_x"] is False
