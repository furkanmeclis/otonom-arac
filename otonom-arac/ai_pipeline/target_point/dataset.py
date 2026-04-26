"""Target-point dataset utilities for legacy tubs and Phase 3 manifests."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from donkeycar.parts.tub_v2 import Tub

logger = logging.getLogger(__name__)

FIXED_1P2M = "fixed_1p2m"
ADAPTIVE_V1 = "adaptive_v1"
MANIFEST_LABEL_MODES = (FIXED_1P2M, ADAPTIVE_V1)


@dataclass(frozen=True)
class TargetPointSample:
    image_path: str
    target_x: float
    target_y: float
    group_id: str
    turn_deg: float = 0.0
    future_steps: int = 0
    lookahead_distance: float = 0.0
    tub_name: str = ""
    record_index: int = 0
    split: str = ""
    track_name: str = ""
    episode_id: str = ""
    frame_index: int = 0
    scenario: str = "nominal"
    label_mode: str = ""
    curvature_score: float = 0.0
    speed_mps: float = 0.0
    teacher_steering: float = float("nan")
    teacher_throttle: float = float("nan")
    lookahead_m: float = 0.0
    cte_m: float = 0.0
    distance_to_centerline_m: float = 0.0
    clean_target_x: float = float("nan")
    clean_target_y: float = float("nan")
    applied_target_x: float = float("nan")
    applied_target_y: float = float("nan")
    driver_source: str = "teacher"
    rollout_event: str = "none"
    rollout_event_id: int = -1
    deviation_active: bool = False
    failure_margin: bool = False

    def resolved_target(self, label_source: str = "clean") -> Tuple[float, float]:
        clean_x = float(self.clean_target_x) if math.isfinite(float(self.clean_target_x)) else float(self.target_x)
        clean_y = float(self.clean_target_y) if math.isfinite(float(self.clean_target_y)) else float(self.target_y)
        applied_x = float(self.applied_target_x) if math.isfinite(float(self.applied_target_x)) else clean_x
        applied_y = float(self.applied_target_y) if math.isfinite(float(self.applied_target_y)) else clean_y

        source = (label_source or "clean").strip().lower()
        if source == "applied":
            return applied_x, applied_y
        if source == "hybrid_recovery_applied":
            if (self.scenario or "nominal") == "recovery":
                return applied_x, applied_y
            return clean_x, clean_y
        return clean_x, clean_y


def is_tub_path(path: Path) -> bool:
    return (path / "manifest.json").exists()


def resolve_tub_paths(tub_paths: str) -> List[str]:
    resolved: List[str] = []
    seen = set()

    for raw_path in (part.strip() for part in (tub_paths or "").split(",")):
        if not raw_path:
            continue

        path = Path(os.path.expanduser(raw_path)).resolve()
        if not path.exists():
            raise ValueError(f"Target-point training path does not exist: {path}")

        child_tubs = sorted(entry.resolve() for entry in path.iterdir() if entry.is_dir() and is_tub_path(entry))
        if is_tub_path(path) and child_tubs:
            candidates = child_tubs
        elif is_tub_path(path):
            candidates = [path]
        else:
            candidates = child_tubs

        if not candidates:
            raise ValueError(
                f"Path '{path}' is not a DonkeyCar tub directory and does not contain tub subdirectories."
            )

        for candidate in candidates:
            candidate_str = candidate.as_posix()
            if candidate_str not in seen:
                seen.add(candidate_str)
                resolved.append(candidate_str)

    if not resolved:
        raise ValueError("No DonkeyCar tub paths were resolved for target-point training.")

    return resolved


def compute_cumulative_distances(positions: np.ndarray) -> np.ndarray:
    if len(positions) == 0:
        return np.array([], dtype=np.float32)

    cumulative = np.zeros(len(positions), dtype=np.float32)
    if len(positions) > 1:
        step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        cumulative[1:] = np.cumsum(step_distances, dtype=np.float32)
    return cumulative


def estimate_heading(positions: np.ndarray, index: int, eps: float = 1e-6) -> Optional[np.ndarray]:
    if len(positions) < 2:
        return None

    prev_index = max(0, index - 1)
    next_index = min(len(positions) - 1, index + 1)
    direction = positions[next_index] - positions[prev_index]
    norm = np.linalg.norm(direction)

    if norm < eps:
        if index + 1 < len(positions):
            direction = positions[index + 1] - positions[index]
            norm = np.linalg.norm(direction)
        if norm < eps and index > 0:
            direction = positions[index] - positions[index - 1]
            norm = np.linalg.norm(direction)

    if norm < eps:
        return None

    return direction / norm


def find_lookahead_index(cumulative_distances: np.ndarray, index: int, lookahead_meters: float) -> Optional[int]:
    if index >= len(cumulative_distances):
        return None

    target_distance = float(cumulative_distances[index]) + float(lookahead_meters)
    future_index = int(np.searchsorted(cumulative_distances, target_distance, side="left"))
    if future_index <= index:
        future_index = index + 1
    if future_index >= len(cumulative_distances):
        return None
    return future_index


def world_to_ego(current_position: np.ndarray, heading: np.ndarray, target_position: np.ndarray) -> Tuple[float, float]:
    delta = target_position - current_position
    right = np.array([heading[1], -heading[0]], dtype=np.float32)
    target_x = float(np.dot(delta, right))
    target_y = float(np.dot(delta, heading))
    return target_x, target_y


def compute_target_point(
    positions: np.ndarray,
    cumulative_distances: np.ndarray,
    index: int,
    lookahead_meters: float,
) -> Optional[Tuple[float, float]]:
    future_index = find_lookahead_index(cumulative_distances, index, lookahead_meters)
    if future_index is None:
        return None

    heading = estimate_heading(positions, index)
    if heading is None:
        return None

    return world_to_ego(positions[index], heading, positions[future_index])


def _load_tub_records(tub_path: str) -> List[dict]:
    tub = None
    try:
        tub = Tub(tub_path, read_only=True)
        records = [dict(record) for record in tub if "__empty__" not in record]
    except (OSError, ValueError) as exc:
        logger.warning("Skipping tub '%s' because it could not be read: %s", tub_path, exc)
        return []
    finally:
        if tub is not None:
            tub.close()

    if not records:
        logger.warning("Skipping tub '%s' because it has no readable records.", tub_path)
        return []
    return records


def _split_records_by_group(records: Sequence[dict], tub_path: str) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    tub_name = Path(tub_path).name

    for record in records:
        group_key = record.get("_session_id") or tub_name
        groups[f"{tub_name}:{group_key}"].append(record)

    for group_key, group_records in groups.items():
        group_records.sort(key=lambda record: (record.get("_timestamp_ms", 0), record.get("_index", 0)))

    return groups


def _require_position_fields(record: dict, tub_path: str) -> None:
    required_fields = ("pos/pos_x", "pos/pos_z", "cam/image_array")
    missing = [field for field in required_fields if field not in record or record[field] is None]
    if missing:
        raise ValueError(
            f"Tub '{tub_path}' is missing required target-point fields {missing}. "
            "Collect fresh simulator data with SIM_RECORD_LOCATION=True."
        )


def _build_samples_for_group(
    records: Sequence[dict],
    tub_path: str,
    group_id: str,
    lookahead_meters: float,
    min_forward: float,
) -> List[TargetPointSample]:
    if len(records) < 2:
        return []

    for record in records:
        _require_position_fields(record, tub_path)

    positions = np.array(
        [[float(record["pos/pos_x"]), float(record["pos/pos_z"])] for record in records],
        dtype=np.float32,
    )
    cumulative_distances = compute_cumulative_distances(positions)
    tub_name = Path(tub_path).name

    samples: List[TargetPointSample] = []
    tub_image_root = Path(tub_path) / "images"

    for index, record in enumerate(records):
        future_index = find_lookahead_index(cumulative_distances, index, lookahead_meters)
        if future_index is None:
            continue

        heading = estimate_heading(positions, index)
        if heading is None:
            continue

        target_x, target_y = world_to_ego(positions[index], heading, positions[future_index])
        if target_y <= float(min_forward):
            continue

        future_heading = estimate_heading(positions, future_index)
        turn_deg = 0.0
        if future_heading is not None:
            cross = heading[0] * future_heading[1] - heading[1] * future_heading[0]
            dot = heading[0] * future_heading[0] + heading[1] * future_heading[1]
            turn_deg = math.degrees(math.atan2(cross, dot))

        image_path = tub_image_root / str(record["cam/image_array"])
        samples.append(
            TargetPointSample(
                image_path=image_path.as_posix(),
                target_x=float(target_x),
                target_y=float(target_y),
                clean_target_x=float(target_x),
                clean_target_y=float(target_y),
                applied_target_x=float(target_x),
                applied_target_y=float(target_y),
                group_id=group_id,
                turn_deg=float(turn_deg),
                future_steps=int(future_index - index),
                lookahead_distance=float(np.linalg.norm(positions[future_index] - positions[index])),
                tub_name=tub_name,
                record_index=int(record.get("_index", index)),
                frame_index=int(record.get("_index", index)),
                episode_id=group_id,
            )
        )

    return samples


def split_samples_by_group(
    samples: Sequence[TargetPointSample],
    train_fraction: float,
    seed: int = 42,
) -> Tuple[List[TargetPointSample], List[TargetPointSample], Dict[str, int]]:
    grouped_samples: Dict[str, List[TargetPointSample]] = defaultdict(list)
    for sample in samples:
        grouped_samples[sample.group_id].append(sample)

    group_ids = list(grouped_samples.keys())
    if not group_ids:
        raise ValueError("No usable target-point samples were generated.")

    if len(group_ids) == 1:
        only_group = grouped_samples[group_ids[0]]
        if len(only_group) < 2:
            raise ValueError(
                "Need at least two usable samples to build train and validation splits for target-point training."
            )

        split_index = max(1, int(len(only_group) * float(train_fraction)))
        split_index = min(split_index, len(only_group) - 1)
        return (
            only_group[:split_index],
            only_group[split_index:],
            {"group_count": 1, "split_strategy": "single_group_temporal"},
        )

    rng = random.Random(seed)
    rng.shuffle(group_ids)

    train_group_count = max(1, int(round(len(group_ids) * float(train_fraction))))
    train_group_count = min(train_group_count, len(group_ids) - 1)
    train_group_ids = set(group_ids[:train_group_count])

    train_samples: List[TargetPointSample] = []
    val_samples: List[TargetPointSample] = []
    for group_id, group_records in grouped_samples.items():
        if group_id in train_group_ids:
            train_samples.extend(group_records)
        else:
            val_samples.extend(group_records)

    if not train_samples or not val_samples:
        raise ValueError("Target-point split produced an empty training or validation set.")

    return train_samples, val_samples, {"group_count": len(group_ids), "split_strategy": "grouped_session"}


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _default_manifest_root(cfg) -> Optional[Path]:
    manifest_root = getattr(cfg, "TARGET_POINT_MANIFEST_ROOT", None)
    if not manifest_root:
        return None
    path = Path(os.path.expanduser(str(manifest_root))).resolve()
    return path if path.exists() else None


def _resolve_manifest_paths(manifest_source: Optional[str], cfg, label_mode: str) -> Tuple[Path, Path, Path]:
    if label_mode not in MANIFEST_LABEL_MODES:
        raise ValueError(f"Unsupported target-point label mode: {label_mode!r}")

    source_path: Optional[Path]
    if manifest_source:
        source_path = Path(os.path.expanduser(str(manifest_source))).resolve()
        if not source_path.exists():
            raise ValueError(f"Manifest source does not exist: {source_path}")
    else:
        source_path = _default_manifest_root(cfg)

    if source_path is None:
        raise ValueError(
            "No manifest source was provided for target-point training and TARGET_POINT_MANIFEST_ROOT is not set."
        )

    if source_path.is_file():
        if source_path.name == "manifest_artifacts.json":
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            manifests = payload.get("sample_manifests", {}).get(label_mode, {})
            train_path = Path(str(manifests.get("train", ""))).resolve()
            val_path = Path(str(manifests.get("val", ""))).resolve()
            if not train_path.exists() or not val_path.exists():
                raise ValueError(f"Manifest artifact file is missing train/val paths for label mode {label_mode!r}.")
            return train_path, val_path, source_path.parent

        expected_train_name = f"samples_train_{label_mode}.jsonl"
        if source_path.name != expected_train_name:
            raise ValueError(
                f"Manifest file '{source_path.name}' is not a supported entrypoint. "
                f"Pass an index directory, manifest_artifacts.json, or {expected_train_name!r}."
            )
        train_path = source_path
        val_path = source_path.with_name(f"samples_val_{label_mode}.jsonl")
        if not val_path.exists():
            raise ValueError(f"Validation manifest was not found next to {train_path}: {val_path}")
        return train_path, val_path, source_path.parent

    index_root = source_path
    if not (index_root / f"samples_train_{label_mode}.jsonl").exists() and (index_root / "index").exists():
        index_root = index_root / "index"

    train_path = index_root / f"samples_train_{label_mode}.jsonl"
    val_path = index_root / f"samples_val_{label_mode}.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise ValueError(
            f"Could not resolve manifest pair for label mode {label_mode!r} under {index_root}. "
            f"Expected {train_path.name} and {val_path.name}."
        )
    return train_path, val_path, index_root


def _sample_from_manifest_row(row: Dict[str, object]) -> TargetPointSample:
    image_path = Path(str(row["image_path"])).resolve()
    speed_mps = max(float(row.get("forward_vel", 0.0)), float(row.get("speed", 0.0)), 0.0)
    episode_id = str(row.get("episode_id", row.get("group_id", row.get("track_name", ""))))
    frame_index = int(row.get("frame_index", row.get("record_index", 0)))
    return TargetPointSample(
        image_path=image_path.as_posix(),
        target_x=float(row["target_x"]),
        target_y=float(row["target_y"]),
        clean_target_x=float(row.get("clean_target_x", row["target_x"])),
        clean_target_y=float(row.get("clean_target_y", row["target_y"])),
        applied_target_x=float(row.get("applied_target_x", row.get("target_x", 0.0))),
        applied_target_y=float(row.get("applied_target_y", row.get("target_y", 0.0))),
        group_id=episode_id,
        turn_deg=float(row.get("delta_heading_2m_deg", 0.0)),
        future_steps=int(row.get("future_steps", 0)),
        lookahead_distance=float(row.get("lookahead_m", row.get("lookahead_distance", 0.0))),
        tub_name=str(row.get("track_name", row.get("map_id", ""))),
        record_index=frame_index,
        split=str(row.get("split", "")),
        track_name=str(row.get("track_name", "")),
        episode_id=episode_id,
        frame_index=frame_index,
        scenario=str(row.get("scenario", "nominal")),
        label_mode=str(row.get("label_mode", row.get("teacher_label_mode", ""))),
        curvature_score=float(row.get("curvature_score", 0.0)),
        speed_mps=float(speed_mps),
        teacher_steering=float(row.get("teacher_steering", float("nan"))),
        teacher_throttle=float(row.get("teacher_throttle", float("nan"))),
        lookahead_m=float(row.get("lookahead_m", row.get("clean_lookahead_m", 0.0))),
        cte_m=float(row.get("cte", 0.0)),
        distance_to_centerline_m=float(row.get("distance_to_centerline_m", 0.0)),
        driver_source=str(row.get("driver_source", "teacher")),
        rollout_event=str(row.get("rollout_event", "none")),
        rollout_event_id=int(row.get("rollout_event_id", -1)),
        deviation_active=bool(row.get("deviation_active", False)),
        failure_margin=bool(row.get("failure_margin", False)),
    )


def _manifest_stats(
    train_rows: Sequence[Dict[str, object]],
    val_rows: Sequence[Dict[str, object]],
    index_root: Path,
    label_mode: str,
) -> Dict[str, object]:
    all_rows = list(train_rows) + list(val_rows)
    scenario_counts = Counter(str(row.get("scenario", "nominal")) for row in all_rows)
    track_counts = Counter(str(row.get("track_name", "")) for row in all_rows)
    return {
        "index_root": index_root.as_posix(),
        "split_strategy": "manifest_fixed_split",
        "label_mode": label_mode,
        "usable_samples": len(all_rows),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "train_recovery_samples": int(sum(1 for row in train_rows if str(row.get("scenario", "nominal")) == "recovery")),
        "val_recovery_samples": int(sum(1 for row in val_rows if str(row.get("scenario", "nominal")) == "recovery")),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "track_counts": dict(sorted(track_counts.items())),
    }


def load_target_point_manifest_splits(
    cfg,
    manifest_source: Optional[str],
    label_mode: str,
) -> Tuple[List[TargetPointSample], List[TargetPointSample], Dict[str, object]]:
    train_path, val_path, index_root = _resolve_manifest_paths(manifest_source, cfg, label_mode)
    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    if not train_rows or not val_rows:
        raise ValueError(
            f"Manifest training split is empty for label mode {label_mode!r}. "
            f"train_rows={len(train_rows)} val_rows={len(val_rows)}"
        )

    train_samples = [_sample_from_manifest_row(row) for row in train_rows]
    val_samples = [_sample_from_manifest_row(row) for row in val_rows]
    return train_samples, val_samples, _manifest_stats(train_rows, val_rows, index_root, label_mode)


def load_target_point_splits(
    cfg,
    tub_paths: Optional[str] = None,
    manifest_source: Optional[str] = None,
    label_mode: str = ADAPTIVE_V1,
) -> Tuple[List[TargetPointSample], List[TargetPointSample], Dict[str, object]]:
    if manifest_source or (not tub_paths and _default_manifest_root(cfg) is not None):
        return load_target_point_manifest_splits(cfg, manifest_source=manifest_source, label_mode=label_mode)

    lookahead_meters = float(getattr(cfg, "TARGET_POINT_LOOKAHEAD_METERS", 1.5))
    min_forward = float(getattr(cfg, "TARGET_POINT_MIN_FORWARD", 0.0))
    train_fraction = float(getattr(cfg, "TRAIN_TEST_SPLIT", 0.8))
    seed = int(getattr(cfg, "TARGET_POINT_SEED", 42))

    all_samples: List[TargetPointSample] = []
    total_records = 0
    total_groups = 0
    skipped_tubs = 0
    resolved_paths = resolve_tub_paths(tub_paths or "")

    for tub_path in resolved_paths:
        records = _load_tub_records(tub_path)
        if not records:
            skipped_tubs += 1
            continue

        total_records += len(records)

        for group_id, group_records in _split_records_by_group(records, tub_path).items():
            total_groups += 1
            all_samples.extend(
                _build_samples_for_group(
                    records=group_records,
                    tub_path=tub_path,
                    group_id=group_id,
                    lookahead_meters=lookahead_meters,
                    min_forward=min_forward,
                )
            )

    if not all_samples:
        raise ValueError(
            "No usable target-point samples were generated after skipping empty or unreadable tubs. "
            "Verify the tub paths and collect fresh simulator data with SIM_RECORD_LOCATION=True."
        )

    train_samples, val_samples, split_info = split_samples_by_group(
        all_samples,
        train_fraction=train_fraction,
        seed=seed,
    )
    stats = {
        "tub_count": len(resolved_paths),
        "skipped_tubs": skipped_tubs,
        "group_count": int(split_info["group_count"]),
        "split_strategy": split_info["split_strategy"],
        "total_records": total_records,
        "usable_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "label_mode": label_mode,
        "track_counts": {},
    }
    return train_samples, val_samples, stats


def load_mixed_splits(
    cfg,
    tub_paths: Optional[str] = None,
    manifest_source: Optional[str] = None,
    label_mode: str = ADAPTIVE_V1,
) -> Tuple[List[TargetPointSample], List[TargetPointSample], Dict[str, object]]:
    """Load sim data + external real-world data with configurable mixing ratio."""
    external_only = bool(getattr(cfg, "TARGET_POINT_EXTERNAL_ONLY", False))
    if external_only:
        train_sim, val_sim = [], []
        stats = {
            "tub_count": 0,
            "skipped_tubs": 0,
            "group_count": 0,
            "split_strategy": "external_only",
            "total_records": 0,
            "usable_samples": 0,
            "train_samples": 0,
            "val_samples": 0,
            "label_mode": label_mode,
            "track_counts": {},
        }
    else:
        # Load primary (sim) data
        train_sim, val_sim, stats = load_target_point_splits(
            cfg, tub_paths=tub_paths, manifest_source=manifest_source, label_mode=label_mode
        )

    external_root = getattr(cfg, "TARGET_POINT_EXTERNAL_ROOT", None)
    if not external_root:
        return train_sim, val_sim, stats

    # Load external data
    from target_point.external_adapter import scan_external_datasets

    lookahead = float(getattr(cfg, "TARGET_POINT_LOOKAHEAD_METERS", 1.0))
    steer_gain = float(getattr(cfg, "TARGET_POINT_STEER_GAIN", 1.0))
    seed = int(getattr(cfg, "TARGET_POINT_SEED", 42))
    ratio = float(getattr(cfg, "TARGET_POINT_EXTERNAL_DATA_RATIO", 0.30))
    excluded_tubs = list(getattr(cfg, "TARGET_POINT_EXTERNAL_EXCLUDED_TUBS", ()))
    min_usable_ratio = float(getattr(cfg, "TARGET_POINT_EXTERNAL_MIN_USABLE_RATIO", 0.0))
    max_missing_image_ratio = float(getattr(cfg, "TARGET_POINT_EXTERNAL_MAX_MISSING_IMAGE_RATIO", 1.0))

    ext_train, ext_val, ext_stats = scan_external_datasets(
        external_root,
        lookahead_y=lookahead,
        steer_gain=steer_gain,
        seed=seed,
        excluded_tubs=excluded_tubs,
        min_usable_ratio=min_usable_ratio,
        max_missing_image_ratio=max_missing_image_ratio,
    )

    # Subsample external data to match the desired ratio
    # ratio = ext / (sim + ext) => ext_count = sim_count * ratio / (1 - ratio)
    if (not external_only) and ratio > 0 and ratio < 1.0:
        target_ext_train = int(len(train_sim) * ratio / (1.0 - ratio))
        target_ext_val = int(len(val_sim) * ratio / (1.0 - ratio))

        rng = random.Random(seed)
        if len(ext_train) > target_ext_train:
            ext_train = rng.sample(ext_train, target_ext_train)
        if len(ext_val) > target_ext_val:
            ext_val = rng.sample(ext_val, target_ext_val)

    if external_only:
        train_combined = ext_train
        val_combined = ext_val
        stats["usable_samples"] = len(train_combined) + len(val_combined)
        stats["train_samples"] = len(train_combined)
        stats["val_samples"] = len(val_combined)
        stats["group_count"] = int(ext_stats.get("total_tubs", 0))
        stats["total_records"] = int(ext_stats.get("total_samples", 0))
        stats["track_counts"] = dict(ext_stats.get("tub_stats", {}))
    else:
        train_combined = train_sim + ext_train
        val_combined = val_sim + ext_val

    stats["external_train"] = len(ext_train)
    stats["external_val"] = len(ext_val)
    stats["external_tubs"] = ext_stats["total_tubs"]
    stats["external_total"] = ext_stats["total_samples"]
    if external_only:
        stats["mix_ratio"] = 1.0
    else:
        stats["mix_ratio"] = len(ext_train) / max(1, len(train_combined))
    stats["external_quality_report"] = ext_stats.get("quality_report", {})

    logger.info(
        "Mixed dataset: %d sim_train + %d ext_train = %d total (%.1f%% external)",
        len(train_sim), len(ext_train), len(train_combined),
        stats["mix_ratio"] * 100,
    )
    return train_combined, val_combined, stats
