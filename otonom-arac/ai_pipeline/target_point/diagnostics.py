"""Diagnostics and visualization helpers for target-point training."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from target_point.controller import target_point_to_controls
from target_point.dataset import TargetPointSample
from target_point.model import preprocess_image

STRAIGHT_THRESHOLD_DEG = 3.0
TURN_THRESHOLD_DEG = 10.0
MAX_TRAIN_DIAGNOSTIC_SAMPLES = 256
MAX_VAL_DIAGNOSTIC_SAMPLES = 2000


def _label_source(cfg) -> str:
    return str(getattr(cfg, "TARGET_POINT_LABEL_SOURCE", "clean")).strip().lower()


def _resolved_target(sample: TargetPointSample, cfg):
    return sample.resolved_target(_label_source(cfg))


def select_train_subset(samples: Sequence[TargetPointSample], seed: int) -> List[TargetPointSample]:
    if len(samples) <= MAX_TRAIN_DIAGNOSTIC_SAMPLES:
        return list(samples)

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    chosen = sorted(indices[:MAX_TRAIN_DIAGNOSTIC_SAMPLES])
    return [samples[index] for index in chosen]


def select_val_subset(samples: Sequence[TargetPointSample], seed: int) -> List[TargetPointSample]:
    if len(samples) <= MAX_VAL_DIAGNOSTIC_SAMPLES:
        return list(samples)

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    chosen = sorted(indices[:MAX_VAL_DIAGNOSTIC_SAMPLES])
    return [samples[index] for index in chosen]


def _load_model_inputs(samples: Sequence[TargetPointSample], cfg):
    images = []
    labels = []
    turn_deg = []
    for sample in samples:
        with Image.open(sample.image_path) as image:
            images.append(preprocess_image(np.asarray(image), cfg))
        labels.append(_resolved_target(sample, cfg))
        turn_deg.append(sample.turn_deg)

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(turn_deg, dtype=np.float32),
    )


def _safe_corrcoef(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(labels) < 2:
        return float("nan")
    if float(np.std(labels)) < 1e-8 or float(np.std(predictions)) < 1e-8:
        return float("nan")
    return float(np.corrcoef(labels, predictions)[0, 1])


def _segment_metrics(labels: np.ndarray, predictions: np.ndarray, turn_deg: np.ndarray) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    masks = {
        "straight": np.abs(turn_deg) < STRAIGHT_THRESHOLD_DEG,
        "turn": np.abs(turn_deg) > TURN_THRESHOLD_DEG,
    }

    for name, mask in masks.items():
        if not np.any(mask):
            results[name] = {"count": 0}
            continue

        errors = predictions[mask] - labels[mask]
        results[name] = {
            "count": int(mask.sum()),
            "label_x_std": float(labels[mask, 0].std()),
            "pred_x_std": float(predictions[mask, 0].std()),
            "label_y_std": float(labels[mask, 1].std()),
            "pred_y_std": float(predictions[mask, 1].std()),
            "mae_x": float(np.abs(errors[:, 0]).mean()),
            "mae_y": float(np.abs(errors[:, 1]).mean()),
        }

    return results


def _controller_stability_metrics(
    samples: Sequence[TargetPointSample],
    predictions: np.ndarray,
    cfg,
) -> Dict[str, float]:
    if len(samples) < 2 or len(predictions) != len(samples):
        return {
            "stability_p95": float("nan"),
            "stability_mean": float("nan"),
            "stability_sample_count": 0,
            "teacher_stability_p95": float("nan"),
        }

    steer_gain = float(getattr(cfg, "TARGET_POINT_STEER_GAIN", 1.0))
    steer_sign = float(getattr(cfg, "TARGET_POINT_STEER_SIGN", 1.0))
    throttle = float(getattr(cfg, "TARGET_POINT_THROTTLE", 0.2))
    min_forward = float(getattr(cfg, "TARGET_POINT_MIN_FORWARD", 0.0))

    ordered = []
    for sample, prediction in zip(samples, predictions):
        predicted_steer, _ = target_point_to_controls(
            target_x=float(prediction[0]),
            target_y=float(prediction[1]),
            steer_gain=steer_gain,
            steer_sign=steer_sign,
            throttle=throttle,
            min_forward=min_forward,
        )
        teacher_steer = float(sample.teacher_steering)
        ordered.append(
            (
                sample.episode_id or sample.group_id,
                int(sample.frame_index),
                float(predicted_steer),
                teacher_steer if math.isfinite(teacher_steer) else float("nan"),
            )
        )

    ordered.sort(key=lambda item: (item[0], item[1]))
    predicted_deltas = []
    teacher_deltas = []
    for previous, current in zip(ordered, ordered[1:]):
        if previous[0] != current[0]:
            continue
        frame_gap = current[1] - previous[1]
        if frame_gap <= 0 or frame_gap > 3:
            continue
        predicted_deltas.append(abs(current[2] - previous[2]))
        if math.isfinite(previous[3]) and math.isfinite(current[3]):
            teacher_deltas.append(abs(current[3] - previous[3]))

    if not predicted_deltas:
        return {
            "stability_p95": float("nan"),
            "stability_mean": float("nan"),
            "stability_sample_count": 0,
            "teacher_stability_p95": float("nan"),
        }

    predicted_delta_array = np.asarray(predicted_deltas, dtype=np.float32)
    teacher_stability_p95 = float("nan")
    if teacher_deltas:
        teacher_stability_p95 = float(np.percentile(np.asarray(teacher_deltas, dtype=np.float32), 95))

    return {
        "stability_p95": float(np.percentile(predicted_delta_array, 95)),
        "stability_mean": float(np.mean(predicted_delta_array)),
        "stability_sample_count": int(predicted_delta_array.size),
        "teacher_stability_p95": teacher_stability_p95,
    }


def summarize_predictions(model, samples: Sequence[TargetPointSample], cfg, split_name: str) -> Dict[str, object]:
    images, labels, turn_deg = _load_model_inputs(samples, cfg)
    predictions = model.predict(images, verbose=0)
    errors = predictions - labels

    label_x_std = float(labels[:, 0].std())
    label_y_std = float(labels[:, 1].std())
    pred_x_std = float(predictions[:, 0].std())
    pred_y_std = float(predictions[:, 1].std())
    stability = _controller_stability_metrics(samples, predictions, cfg)

    return {
        "split_name": split_name,
        "sample_count": len(samples),
        "label_x_std": label_x_std,
        "pred_x_std": pred_x_std,
        "pred_x_std_ratio": float(pred_x_std / label_x_std) if label_x_std > 1e-8 else float("nan"),
        "label_y_std": label_y_std,
        "pred_y_std": pred_y_std,
        "pred_y_std_ratio": float(pred_y_std / label_y_std) if label_y_std > 1e-8 else float("nan"),
        "label_x_mean": float(labels[:, 0].mean()),
        "pred_x_mean": float(predictions[:, 0].mean()),
        "label_y_mean": float(labels[:, 1].mean()),
        "pred_y_mean": float(predictions[:, 1].mean()),
        "mae_x": float(np.abs(errors[:, 0]).mean()),
        "mae_y": float(np.abs(errors[:, 1]).mean()),
        "corr_x": _safe_corrcoef(labels[:, 0], predictions[:, 0]),
        "corr_y": _safe_corrcoef(labels[:, 1], predictions[:, 1]),
        "stability_p95": float(stability["stability_p95"]),
        "stability_mean": float(stability["stability_mean"]),
        "stability_sample_count": int(stability["stability_sample_count"]),
        "teacher_stability_p95": float(stability["teacher_stability_p95"]),
        "segment_metrics": _segment_metrics(labels, predictions, turn_deg),
    }


def summarize_predictions_by_track(model, samples: Sequence[TargetPointSample], cfg, split_name_prefix: str) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, List[TargetPointSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.track_name or "unknown"].append(sample)

    summaries: Dict[str, Dict[str, object]] = {}
    for track_name, track_samples in sorted(grouped.items()):
        summaries[track_name] = summarize_predictions(
            model,
            track_samples,
            cfg,
            split_name=f"{split_name_prefix}:{track_name}",
        )
    return summaries


def _gate_check(value: float, threshold: float, op: str) -> bool:
    if not math.isfinite(value):
        return False
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold
    raise ValueError(f"Unsupported gate operator: {op}")


def evaluate_collapse_gate(train_metrics: Dict[str, object], val_metrics: Dict[str, object]) -> Dict[str, object]:
    turn_metrics = val_metrics["segment_metrics"].get("turn", {})
    checks = {
        "val_pred_x_std_ratio": _gate_check(float(val_metrics["pred_x_std_ratio"]), 0.25, ">="),
        "train_pred_x_std_ratio": _gate_check(float(train_metrics["pred_x_std_ratio"]), 0.20, ">="),
        "val_corr_x": _gate_check(float(val_metrics["corr_x"]), 0.40, ">"),
        "turn_mae_x": _gate_check(float(turn_metrics.get("mae_x", float("inf"))), 0.20, "<"),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _spread(items: Sequence[TargetPointSample], count: int) -> List[TargetPointSample]:
    if len(items) <= count:
        return list(items)
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[int(index)] for index in indices]


def _select_contact_sheet_samples(samples: Sequence[TargetPointSample]) -> List[TargetPointSample]:
    straight = [sample for sample in samples if abs(sample.turn_deg) < STRAIGHT_THRESHOLD_DEG]
    left_turn = [sample for sample in samples if sample.turn_deg > TURN_THRESHOLD_DEG]
    right_turn = [sample for sample in samples if sample.turn_deg < -TURN_THRESHOLD_DEG]

    selected = _spread(straight, 10) + _spread(left_turn, 10) + _spread(right_turn, 10)
    if len(selected) >= 30:
        return selected[:30]

    seen = {sample.image_path for sample in selected}
    for sample in samples:
        if sample.image_path in seen:
            continue
        selected.append(sample)
        seen.add(sample.image_path)
        if len(selected) == 30:
            break
    return selected


def _sample_category(sample: TargetPointSample) -> str:
    if sample.turn_deg > TURN_THRESHOLD_DEG:
        return "left_turn"
    if sample.turn_deg < -TURN_THRESHOLD_DEG:
        return "right_turn"
    return "straight"


def write_contact_sheet(samples: Sequence[TargetPointSample], output_dir: str) -> Dict[str, object]:
    selected = _select_contact_sheet_samples(samples)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()
    colors = {
        "straight": (60, 220, 60),
        "left_turn": (255, 120, 80),
        "right_turn": (80, 160, 255),
    }
    max_x = max(1.0, max(abs(_resolved_target(sample, None)[0]) for sample in selected) * 1.1) if selected else 1.0
    max_y = max(1.0, max(_resolved_target(sample, None)[1] for sample in selected) * 1.1) if selected else 1.0

    rendered = []
    for index, sample in enumerate(selected, start=1):
        category = _sample_category(sample)
        image = Image.open(sample.image_path).convert("RGB").resize((320, 240))
        draw = ImageDraw.Draw(image)
        color = colors[category]
        draw.rectangle((0, 0, 319, 28), fill=(0, 0, 0))
        target_x, target_y = sample.resolved_target("clean")
        summary = (
            f"{index:02d} {category}  tub={sample.tub_name}  idx={sample.record_index}  "
            f"tx={target_x:.2f} ty={target_y:.2f}"
        )
        draw.text((6, 8), summary, fill=(255, 255, 255), font=font)

        inset = (215, 118, 314, 236)
        draw.rounded_rectangle(inset, radius=8, fill=(20, 20, 20), outline=color, width=2)
        x0, y0, x1, y1 = inset
        origin = ((x0 + x1) // 2, y1 - 10)
        draw.line((origin[0], y0 + 8, origin[0], y1 - 8), fill=(100, 100, 100), width=1)
        draw.line((x0 + 8, origin[1], x1 - 8, origin[1]), fill=(100, 100, 100), width=1)

        px = origin[0] + int((target_x / max_x) * ((x1 - x0) * 0.38))
        py = origin[1] - int((target_y / max_y) * ((y1 - y0) * 0.75))
        draw.line((origin[0], origin[1], px, py), fill=color, width=3)
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
        draw.text((x0 + 5, y0 + 4), "ego", fill=(220, 220, 220), font=font)

        file_name = f"sample_{index:02d}_{category}.jpg"
        file_path = output_path / file_name
        image.save(file_path, quality=90)
        rendered.append((image.copy(), sample, file_name, category))

    contact_sheet = Image.new("RGB", (5 * 320, 6 * 240), (15, 15, 15))
    for index, (image, _, _, _) in enumerate(rendered):
        col = index % 5
        row = index // 5
        contact_sheet.paste(image, (col * 320, row * 240))

    contact_sheet_path = output_path / "label_contact_sheet.jpg"
    contact_sheet.save(contact_sheet_path, quality=90)

    csv_path = output_path / "label_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "category",
                "tub_name",
                "record_index",
                "target_x",
                "target_y",
                "turn_deg",
                "future_steps",
                "lookahead_distance",
                "image_path",
            ],
        )
        writer.writeheader()
        for sample_image, sample, file_name, category in rendered:
            del sample_image
            target_x, target_y = sample.resolved_target("clean")
            writer.writerow(
                {
                    "file": file_name,
                    "category": category,
                    "tub_name": sample.tub_name,
                    "record_index": sample.record_index,
                    "target_x": f"{target_x:.6f}",
                    "target_y": f"{target_y:.6f}",
                    "turn_deg": f"{sample.turn_deg:.6f}",
                    "future_steps": sample.future_steps,
                    "lookahead_distance": f"{sample.lookahead_distance:.6f}",
                    "image_path": sample.image_path,
                }
            )

    return {
        "contact_sheet_path": contact_sheet_path.as_posix(),
        "csv_path": csv_path.as_posix(),
        "selected_count": len(rendered),
    }


def write_diagnostics_report(
    output_dir: str,
    train_metrics: Dict[str, object],
    val_metrics: Dict[str, object],
    gate: Dict[str, object],
    visualization: Dict[str, object],
) -> str:
    payload = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "collapse_gate": gate,
        "visualization": visualization,
    }
    path = Path(output_dir) / "diagnostics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path.as_posix()
