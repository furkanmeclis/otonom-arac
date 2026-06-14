"""Faz 2/3 ham bölümleri: indeksleme, filtreleme, dengeleme ve manifest üretimi.

Bu modül, toplanan ham sürüş bölümlerini (episodes) eğitime hazır JSONL
manifest'lerine dönüştürür. build_target_point_labels.py'nin --raw-root(s) ile
çağırdığı asıl iş burada (build_phase2_manifests) yapılır.

İş akışı:
  1. Bölümleri keşfet, pist haritalarını yükle.
  2. Her kareye iki etiket modunda (fixed_1p2m, adaptive_v1) target nokta hesapla.
  3. Bozuk/geçersiz kareleri FİLTRELE (sonsuz değer, eksik alan, yola çok uzak...).
  4. DENGELE: kurtarma (recovery) ve düz sürüş oranını ayarla, rollout payını
     sınırla (bir senaryo veriyi domine etmesin diye).
  5. train/val manifest'lerini + lookahead/filtre raporlarını yaz.

Çıktı: her etiket modu ve split için ayrı .jsonl + teşhis raporları.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from target_point.collector import discover_track_maps
from target_point.teacher_policy import (
    ADAPTIVE_V1,
    FIXED_1P2M,
    LOOKAHEAD_CURVATURE_BINS,
    LOOKAHEAD_SPEED_BINS,
    materialize_label_from_state,
)
from target_point.track_map import TrackMapArtifact, load_track_map


MANIFEST_LABEL_MODES = (FIXED_1P2M, ADAPTIVE_V1)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _wrap_delta_deg(current: float, previous: float) -> float:
    return ((float(current) - float(previous) + 180.0) % 360.0) - 180.0


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


def _numeric_summary(values: Iterable[float]) -> Dict[str, float | None]:
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


def _summary_by_group(rows: Sequence[Dict[str, object]], key_name: str) -> Dict[str, Dict[str, float | None]]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        grouped.setdefault(str(row[key_name]), []).append(float(row["lookahead_m"]))
    return {key: _numeric_summary(values) for key, values in sorted(grouped.items())}


def build_lookahead_report(label_mode: str, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    with_speed_bins = []
    for row in rows:
        with_speed_bins.append(
            {
                **row,
                "speed_bin": _bucket_label(float(max(float(row["forward_vel"]), float(row["speed"]), 0.0)), LOOKAHEAD_SPEED_BINS),
                "curvature_bin": _bucket_label(float(row["curvature_score"]), LOOKAHEAD_CURVATURE_BINS),
            }
        )
    return {
        "label_mode": label_mode,
        "overall": _numeric_summary(float(row["lookahead_m"]) for row in with_speed_bins),
        "by_split": _summary_by_group(with_speed_bins, "split"),
        "by_track": _summary_by_group(with_speed_bins, "track_name"),
        "by_scenario": _summary_by_group(with_speed_bins, "scenario"),
        "by_speed_bin": _summary_by_group(with_speed_bins, "speed_bin"),
        "by_curvature_bin": _summary_by_group(with_speed_bins, "curvature_bin"),
    }


def _increment_filter_count(report: Dict[str, object], reason: str, split: str, track_name: str, scenario: str) -> None:
    reasons = report.setdefault("rejections", {})
    entry = reasons.setdefault(
        reason,
        {"count": 0, "by_split": {}, "by_track": {}, "by_scenario": {}},
    )
    entry["count"] += 1
    entry["by_split"][split] = entry["by_split"].get(split, 0) + 1
    entry["by_track"][track_name] = entry["by_track"].get(track_name, 0) + 1
    entry["by_scenario"][scenario] = entry["by_scenario"].get(scenario, 0) + 1


def _is_finite_value(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _base_filter_reasons(rows: Sequence[Dict[str, object]]) -> List[set[str]]:
    reasons: List[set[str]] = [set() for _ in rows]
    for index, row in enumerate(rows):
        if index < 15:
            reasons[index].add("warmup")

        for key in (
            "pos_x",
            "pos_y",
            "pos_z",
            "yaw_deg",
            "cte",
            "speed",
            "forward_vel",
            "map_distance_m",
            "curvature_score",
            "clean_target_x",
            "clean_target_y",
            "applied_target_x",
            "applied_target_y",
            "teacher_steering",
            "teacher_throttle",
        ):
            if not _is_finite_value(row.get(key)):
                reasons[index].add("nonfinite")
                break

        if str(row.get("hit", "none")) != "none":
            reasons[index].add("hit")
        if bool(row.get("done", False)):
            reasons[index].add("episode_done")
        centerline_distance = float(row.get("distance_to_centerline_m", 0.0))
        if centerline_distance > 1.10:
            reasons[index].add("off_track")

        if index > 0:
            previous = rows[index - 1]
            dx = float(row["pos_x"]) - float(previous["pos_x"])
            dz = float(row["pos_z"]) - float(previous["pos_z"])
            if math.hypot(dx, dz) > 0.75:
                reasons[index].add("position_jump")

            yaw_jump = abs(_wrap_delta_deg(float(row["yaw_deg"]), float(previous["yaw_deg"])))
            if yaw_jump > 20.0:
                reasons[index].add("yaw_jump")

    stall_start = None
    for index, row in enumerate(rows):
        speed_mps = max(float(row["forward_vel"]), float(row["speed"]), 0.0)
        if str(row["scenario"]) == "nominal" and speed_mps < 0.05:
            if stall_start is None:
                stall_start = index
        else:
            if stall_start is not None and (index - stall_start) > 10:
                for stalled in range(stall_start, index):
                    reasons[stalled].add("nominal_stall")
            stall_start = None
    if stall_start is not None and (len(rows) - stall_start) > 10:
        for stalled in range(stall_start, len(rows)):
            reasons[stalled].add("nominal_stall")

    index = 0
    while index < len(rows):
        if str(rows[index]["scenario"]) != "recovery":
            index += 1
            continue
        start = index
        while index < len(rows) and str(rows[index]["scenario"]) == "recovery":
            index += 1
        end = index
        success = False
        future_limit = min(len(rows), end + 40)
        for future_index in range(start, future_limit):
            future_row = rows[future_index]
            if str(future_row.get("hit", "none")) == "none" and abs(float(future_row["cte"])) <= 0.30:
                success = True
                break
        if not success:
            for failed in range(start, end):
                reasons[failed].add("failed_recovery")

    return reasons


def _normalize_raw_roots(raw_root: str | Path | Sequence[str | Path]) -> List[Path]:
    if isinstance(raw_root, (str, Path)):
        candidates = [part.strip() for part in str(raw_root).split(",") if part.strip()]
        if len(candidates) == 1 and str(raw_root).strip() and "," not in str(raw_root):
            return [Path(raw_root).resolve()]
        return [Path(candidate).resolve() for candidate in candidates]
    return [Path(candidate).resolve() for candidate in raw_root]


def _discover_episodes(raw_root: str | Path | Sequence[str | Path]) -> List[Path]:
    episodes: List[Path] = []
    for root in _normalize_raw_roots(raw_root):
        episodes.extend(path.parent for path in root.glob("*/*/*/episode_metadata.json"))
    return sorted(set(episodes))


def _load_map_cache(maps_root: str | Path, episode_metadata_rows: Sequence[Dict[str, object]]) -> Dict[str, TrackMapArtifact]:
    maps_by_track = discover_track_maps(maps_root)
    cache: Dict[str, TrackMapArtifact] = {}
    for metadata in episode_metadata_rows:
        map_id = str(metadata["map_id"])
        if map_id in cache:
            continue
        map_dir = Path(str(metadata.get("map_dir", "")))
        if not map_dir.exists():
            track_name = str(metadata["track_name"])
            if track_name not in maps_by_track:
                raise FileNotFoundError(f"No Phase 1 map found for track {track_name!r}")
            map_dir = maps_by_track[track_name]
        cache[map_id] = load_track_map(map_dir)
    return cache


def _scenario_name(row: Dict[str, object]) -> str:
    return "recovery" if str(row.get("scenario", "nominal")) == "recovery" else "nominal"


def _scenario_counts(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    nominal = 0
    recovery = 0
    for row in rows:
        if _scenario_name(row) == "recovery":
            recovery += 1
        else:
            nominal += 1
    total = nominal + recovery
    recovery_ratio = (float(recovery) / float(total)) if total else 0.0
    return {
        "nominal": int(nominal),
        "recovery": int(recovery),
        "total": int(total),
        "recovery_ratio": float(recovery_ratio),
    }


def _source_counts(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        source = str(row.get("driver_source", "teacher") or "teacher")
        entry = counts.setdefault(source, {"total": 0, "nominal": 0, "recovery": 0})
        entry["total"] += 1
        entry[_scenario_name(row)] += 1
    return counts


def _stable_row_hash(row: Dict[str, object]) -> str:
    payload = "|".join(
        (
            str(row.get("split", "")),
            str(row.get("track_name", "")),
            str(row.get("episode_id", "")),
            str(int(row.get("frame_index", 0))),
            str(row.get("sample_id", "")),
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _sorted_rows_for_sampling(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            _stable_row_hash(row),
            str(row.get("track_name", "")),
            str(row.get("episode_id", "")),
            int(row.get("frame_index", 0)),
        ),
    )


def _downsample_rows(rows: Sequence[Dict[str, object]], keep_count: int) -> List[Dict[str, object]]:
    if keep_count >= len(rows):
        return list(rows)
    if keep_count <= 0:
        return []

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["track_name"]), []).append(row)

    total_rows = len(rows)
    allocations: Dict[str, int] = {}
    fractional: List[tuple[float, str]] = []
    allocated = 0
    for track_name, track_rows in grouped.items():
        ideal = (float(len(track_rows)) / float(total_rows)) * float(keep_count)
        base = min(len(track_rows), int(math.floor(ideal)))
        allocations[track_name] = base
        allocated += base
        fractional.append((ideal - float(base), track_name))

    remaining = int(keep_count - allocated)
    for _, track_name in sorted(fractional, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        capacity = len(grouped[track_name]) - allocations[track_name]
        if capacity <= 0:
            continue
        allocations[track_name] += 1
        remaining -= 1

    if remaining > 0:
        for track_name in sorted(grouped):
            if remaining <= 0:
                break
            capacity = len(grouped[track_name]) - allocations[track_name]
            if capacity <= 0:
                continue
            take = min(capacity, remaining)
            allocations[track_name] += take
            remaining -= take

    selected: List[Dict[str, object]] = []
    for track_name in sorted(grouped):
        keep = allocations.get(track_name, 0)
        if keep <= 0:
            continue
        selected.extend(_sorted_rows_for_sampling(grouped[track_name])[:keep])
    return selected


def _apply_rollout_caps(
    rows: Sequence[Dict[str, object]],
    rollout_max_share: float | None,
    generated_roads_rollout_max_share: float | None,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    report = {
        "status": "skipped",
        "rollout_max_share": rollout_max_share,
        "generated_roads_rollout_max_share": generated_roads_rollout_max_share,
        "pre_counts": _source_counts(rows),
    }
    if not rows:
        report["post_counts"] = report["pre_counts"]
        return list(rows), report

    teacher_rows = [row for row in rows if str(row.get("driver_source", "teacher")) != "rollout"]
    rollout_rows = [row for row in rows if str(row.get("driver_source", "teacher")) == "rollout"]
    if not rollout_rows:
        report["status"] = "no_rollout_rows"
        report["post_counts"] = report["pre_counts"]
        return list(rows), report

    kept_rollout = list(rollout_rows)
    if rollout_max_share is not None:
        if not (0.0 < float(rollout_max_share) < 1.0):
            raise ValueError(f"rollout_max_share must be between 0 and 1, got {rollout_max_share!r}")
        max_rollout_keep = int(
            math.floor((float(rollout_max_share) * float(len(teacher_rows))) / max(1e-6, 1.0 - float(rollout_max_share)))
        )
        max_rollout_keep = max(0, min(max_rollout_keep, len(rollout_rows)))
        kept_rollout = _downsample_rows(kept_rollout, max_rollout_keep)
        report["status"] = "rollout_capped"
        report["max_rollout_keep"] = int(max_rollout_keep)

    if generated_roads_rollout_max_share is not None and kept_rollout:
        if not (0.0 <= float(generated_roads_rollout_max_share) <= 1.0):
            raise ValueError(
                f"generated_roads_rollout_max_share must be between 0 and 1, got {generated_roads_rollout_max_share!r}"
            )
        roads_rows = [row for row in kept_rollout if str(row.get("track_name", "")) == "donkey-generated-roads-v0"]
        other_rows = [row for row in kept_rollout if str(row.get("track_name", "")) != "donkey-generated-roads-v0"]
        allowed_roads = int(math.floor(float(generated_roads_rollout_max_share) * float(len(kept_rollout))))
        allowed_roads = max(0, min(allowed_roads, len(roads_rows)))
        if len(roads_rows) > allowed_roads:
            roads_rows = _downsample_rows(roads_rows, allowed_roads)
            kept_rollout = other_rows + roads_rows
            report["status"] = "roads_rollout_capped" if report["status"] == "skipped" else f"{report['status']}_and_roads_capped"
            report["max_generated_roads_rollout_keep"] = int(allowed_roads)

    capped_rows = sorted(teacher_rows + kept_rollout, key=lambda row: (str(row["track_name"]), str(row["episode_id"]), int(row["frame_index"])))
    report["post_counts"] = _source_counts(capped_rows)
    return capped_rows, report


def _balanced_majority_keep(minority_count: int, majority_available: int, target_ratio: float, recovery_is_minority: bool) -> int:
    if minority_count <= 0 or majority_available <= 0:
        return 0

    if recovery_is_minority:
        ideal = float(minority_count) * (1.0 - float(target_ratio)) / float(target_ratio)
        candidates = {int(math.floor(ideal)), int(math.ceil(ideal))}
        best_keep = None
        best_error = None
        for candidate in candidates:
            candidate = max(0, min(int(candidate), int(majority_available)))
            total = minority_count + candidate
            if total <= 0:
                continue
            achieved = float(minority_count) / float(total)
            error = abs(achieved - float(target_ratio))
            if best_keep is None or error < best_error or (error == best_error and candidate > best_keep):
                best_keep = candidate
                best_error = error
        return int(best_keep if best_keep is not None else min(majority_available, minority_count))

    ideal = float(minority_count) * float(target_ratio) / (1.0 - float(target_ratio))
    candidates = {int(math.floor(ideal)), int(math.ceil(ideal))}
    best_keep = None
    best_error = None
    for candidate in candidates:
        candidate = max(0, min(int(candidate), int(majority_available)))
        total = minority_count + candidate
        if total <= 0:
            continue
        achieved = float(candidate) / float(total)
        error = abs(achieved - float(target_ratio))
        if best_keep is None or error < best_error or (error == best_error and candidate > best_keep):
            best_keep = candidate
            best_error = error
    return int(best_keep if best_keep is not None else min(majority_available, minority_count))


def _balance_rows(rows: Sequence[Dict[str, object]], target_recovery_ratio: float | None) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    """Senaryo dağılımını dengeler (örn. recovery oranını hedefe çeker).
    Çoğunluk senaryosunu (genelde düz sürüş) örnekleyerek azaltır ki model
    sadece düz gitmeyi öğrenmesin. Döner: (dengelenmiş satırlar, öncesi/sonrası rapor)."""
    pre_counts = _scenario_counts(rows)
    report: Dict[str, object] = {
        "pre_balance": pre_counts,
        "target_recovery_ratio": None if target_recovery_ratio is None else float(target_recovery_ratio),
    }
    if not rows or target_recovery_ratio is None:
        report["post_balance"] = pre_counts
        report["status"] = "skipped"
        return list(rows), report

    if not (0.0 < float(target_recovery_ratio) < 1.0):
        raise ValueError(f"target_recovery_ratio must be between 0 and 1, got {target_recovery_ratio!r}")

    nominal_rows = [row for row in rows if _scenario_name(row) == "nominal"]
    recovery_rows = [row for row in rows if _scenario_name(row) == "recovery"]
    if not nominal_rows or not recovery_rows:
        report["post_balance"] = pre_counts
        report["status"] = "skipped_missing_class"
        return list(rows), report

    current_ratio = float(pre_counts["recovery_ratio"])
    if abs(current_ratio - float(target_recovery_ratio)) <= 1e-9:
        report["post_balance"] = pre_counts
        report["status"] = "already_balanced"
        return list(rows), report

    if current_ratio < float(target_recovery_ratio):
        keep_recovery = len(recovery_rows)
        keep_nominal = _balanced_majority_keep(
            minority_count=len(recovery_rows),
            majority_available=len(nominal_rows),
            target_ratio=float(target_recovery_ratio),
            recovery_is_minority=True,
        )
    else:
        keep_nominal = len(nominal_rows)
        keep_recovery = _balanced_majority_keep(
            minority_count=len(nominal_rows),
            majority_available=len(recovery_rows),
            target_ratio=float(target_recovery_ratio),
            recovery_is_minority=False,
        )

    balanced_rows = _downsample_rows(nominal_rows, keep_nominal) + _downsample_rows(recovery_rows, keep_recovery)
    balanced_rows = sorted(balanced_rows, key=lambda row: (str(row["track_name"]), str(row["episode_id"]), int(row["frame_index"])))
    report["post_balance"] = _scenario_counts(balanced_rows)
    report["status"] = "balanced"
    return balanced_rows, report


def build_phase2_manifests(
    raw_root: str | Path | Sequence[str | Path],
    maps_root: str | Path,
    index_root: str | Path,
    target_recovery_ratio: float | None = None,
    rollout_max_share: float | None = None,
    generated_roads_rollout_max_share: float | None = None,
) -> Dict[str, object]:
    """Ham bölümleri eğitime hazır JSONL manifest'lerine çevirir (ana giriş).

    raw_root: ham bölüm klasör(ler)i (phase2/phase3/rollout birleştirilebilir).
    maps_root: etiket hesabı için pist haritalarının kökü.
    index_root: çıktının yazılacağı klasör.
    *_ratio / *_share: senaryo dengeleme ve rollout payı sınırları.
    Döner: üretilen manifest yolları + rapor yollarını içeren sözlük. Karşılığı
    komut: build_target_point_labels.py --raw-root(s) ... --output-dir ..."""
    raw_roots = _normalize_raw_roots(raw_root)
    index_root = Path(index_root).resolve()
    index_root.mkdir(parents=True, exist_ok=True)

    episode_dirs = _discover_episodes(raw_roots)
    if not episode_dirs:
        raise FileNotFoundError(f"No raw episodes found under {', '.join(str(root) for root in raw_roots)}")

    episode_metadata_rows = [
        json.loads((episode_dir / "episode_metadata.json").read_text(encoding="utf-8"))
        for episode_dir in episode_dirs
    ]
    map_cache = _load_map_cache(maps_root, episode_metadata_rows)

    episode_manifest_rows: List[Dict[str, object]] = []
    rows_by_mode_and_split: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    base_filter_report: Dict[str, object] = {"rejections": {}}
    mode_filter_reports: Dict[str, Dict[str, object]] = {mode: {"rejections": {}} for mode in MANIFEST_LABEL_MODES}

    for metadata in sorted(episode_metadata_rows, key=lambda row: (row["split"], row["track_name"], row["episode_id"])):
        episode_dir = Path(str(metadata["raw_samples_jsonl"])).resolve().parent
        episode_dataset_root = episode_dir.parents[3]
        raw_samples = _read_jsonl(episode_dir / "samples_raw.jsonl")
        split = str(metadata["split"])
        track_name = str(metadata["track_name"])
        episode_id = str(metadata["episode_id"])
        map_id = str(metadata["map_id"])
        track_map = map_cache[map_id]

        rejection_sets = _base_filter_reasons(raw_samples)
        retained_base = [index for index, reasons in enumerate(rejection_sets) if not reasons]
        for row, reasons in zip(raw_samples, rejection_sets):
            for reason in reasons:
                _increment_filter_count(base_filter_report, reason, split, track_name, str(row["scenario"]))

        episode_manifest_rows.append(
            {
                **metadata,
                "episode_dir": str(episode_dir),
                "raw_sample_count": len(raw_samples),
                "base_retained_count": len(retained_base),
            }
        )

        for label_mode in MANIFEST_LABEL_MODES:
            retained_rows = rows_by_mode_and_split.setdefault((label_mode, split), [])
            for index in retained_base:
                row = raw_samples[index]
                label = materialize_label_from_state(
                    track_map=track_map,
                    label_mode=label_mode,
                    map_distance_m=float(row["map_distance_m"]),
                    current_position_xyz=(float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])),
                    yaw_deg=float(row["yaw_deg"]),
                    speed_mps=max(float(row["forward_vel"]), float(row["speed"]), 0.0),
                    curvature_score=float(row["curvature_score"]),
                    allow_tail_clamp=False,
                )
                if label is None:
                    _increment_filter_count(mode_filter_reports[label_mode], "label_out_of_range", split, track_name, str(row["scenario"]))
                    continue

                retained_rows.append(
                    {
                        **row,
                        "episode_id": episode_id,
                        "image_path": str((episode_dataset_root / str(row["image_rel_path"])).resolve()),
                        "label_mode": label_mode,
                        "lookahead_m": float(label.lookahead_m),
                        "target_x": float(label.target_x),
                        "target_y": float(label.target_y),
                    }
                )

    episodes_path = index_root / "episodes.jsonl"
    _write_jsonl(episodes_path, episode_manifest_rows)

    output_manifest: Dict[str, object] = {
        "episodes_jsonl": str(episodes_path),
        "raw_roots": [str(root) for root in raw_roots],
        "sample_manifests": {},
        "reports": {},
    }
    all_mode_rows: Dict[str, List[Dict[str, object]]] = {mode: [] for mode in MANIFEST_LABEL_MODES}
    balance_report: Dict[str, object] = {"target_recovery_ratio": target_recovery_ratio, "by_mode": {}}
    bootstrap_report: Dict[str, object] = {
        "rollout_max_share": rollout_max_share,
        "generated_roads_rollout_max_share": generated_roads_rollout_max_share,
        "by_mode": {},
    }

    for (label_mode, split), rows in sorted(rows_by_mode_and_split.items()):
        rows = sorted(rows, key=lambda row: (row["track_name"], row["episode_id"], row["frame_index"]))
        capped_rows = rows
        cap_info = {"status": "skipped", "pre_counts": _source_counts(rows), "post_counts": _source_counts(rows)}
        if split == "train":
            capped_rows, cap_info = _apply_rollout_caps(
                rows,
                rollout_max_share=rollout_max_share,
                generated_roads_rollout_max_share=generated_roads_rollout_max_share,
            )
        bootstrap_report["by_mode"].setdefault(label_mode, {})[split] = cap_info
        balanced_rows, balance_info = _balance_rows(capped_rows, target_recovery_ratio=target_recovery_ratio)
        manifest_path = index_root / f"samples_{split}_{label_mode}.jsonl"
        _write_jsonl(manifest_path, balanced_rows)
        output_manifest["sample_manifests"].setdefault(label_mode, {})[split] = str(manifest_path)
        all_mode_rows[label_mode].extend(balanced_rows)
        balance_report["by_mode"].setdefault(label_mode, {})[split] = balance_info

    filter_report = {
        "episodes_jsonl": str(episodes_path),
        "raw_episode_count": len(episode_manifest_rows),
        "base_filters": base_filter_report,
        "mode_filters": mode_filter_reports,
        "retained_counts_pre_balance": {
            label_mode: {
                split: len(rows_by_mode_and_split.get((label_mode, split), []))
                for split in sorted({key[1] for key in rows_by_mode_and_split.keys() if key[0] == label_mode})
            }
            for label_mode in MANIFEST_LABEL_MODES
        },
    }
    filter_report_path = index_root / "filter_report.json"
    filter_report_path.write_text(json.dumps(filter_report, indent=2, sort_keys=True), encoding="utf-8")
    output_manifest["reports"]["filter_report_json"] = str(filter_report_path)

    for label_mode, rows in all_mode_rows.items():
        lookahead_report = build_lookahead_report(label_mode, rows)
        report_path = index_root / f"lookahead_report_{label_mode}.json"
        report_path.write_text(json.dumps(lookahead_report, indent=2, sort_keys=True), encoding="utf-8")
        output_manifest["reports"][f"lookahead_report_{label_mode}"] = str(report_path)

    balance_report_path = index_root / "balance_report.json"
    balance_report_path.write_text(json.dumps(balance_report, indent=2, sort_keys=True), encoding="utf-8")
    output_manifest["reports"]["balance_report_json"] = str(balance_report_path)

    bootstrap_report_path = index_root / "bootstrap_report.json"
    bootstrap_report_path.write_text(json.dumps(bootstrap_report, indent=2, sort_keys=True), encoding="utf-8")
    output_manifest["reports"]["bootstrap_report_json"] = str(bootstrap_report_path)

    manifest_artifacts_path = index_root / "manifest_artifacts.json"
    manifest_artifacts_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True), encoding="utf-8")
    output_manifest["manifest_artifacts_json"] = str(manifest_artifacts_path)
    return output_manifest
