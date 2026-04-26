#!/usr/bin/env python3
"""Materialize Phase 1 map labels or Phase 2/3 dataset manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import donkeycar as dk

from target_point.collector import default_maps_root
from target_point.manifest import build_phase2_manifests
from target_point.teacher_policy import (
    ADAPTIVE_V1,
    FIXED_1P2M,
    label_records_to_rows,
    materialize_dual_label_modes,
    materialize_label_mode,
    summarize_lookahead_stats,
)
from target_point.track_map import load_track_map


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulationconfig", default="simulationconfig.py", help="Config file to load for dataset defaults. [default: simulationconfig.py]")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--map-dir", help="Directory containing metadata.json, raw_trace.csv, and centerline.csv.")
    mode_group.add_argument("--raw-root", help="Phase 2 raw dataset root, typically <dataset_root>/raw.")
    mode_group.add_argument(
        "--raw-roots",
        help="Comma-separated raw dataset roots to merge, e.g. phase3 raw plus rollout raw.",
    )
    parser.add_argument(
        "--label-modes",
        default="both",
        choices=("both", FIXED_1P2M, ADAPTIVE_V1),
        help="Which Phase 1 map label mode(s) to materialize.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to --map-dir for Phase 1 and <raw_root>/../index for dataset manifests.",
    )
    parser.add_argument("--maps-root", default=None, help="Phase 1 maps root for dataset manifest generation.")
    parser.add_argument(
        "--target-recovery-ratio",
        type=float,
        default=None,
        help="Optional final recovery ratio for manifest balancing, e.g. 0.30.",
    )
    parser.add_argument(
        "--rollout-max-share",
        type=float,
        default=None,
        help="Optional max rollout share in the merged train manifest, e.g. 0.35.",
    )
    parser.add_argument(
        "--generated-roads-rollout-max-share",
        type=float,
        default=None,
        help="Optional max share of rollout rows from donkey-generated-roads-v0 within the rollout subset.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty label file: {path}")

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    if args.map_dir:
        map_dir = Path(args.map_dir).resolve()
        output_dir = Path(args.output_dir).resolve() if args.output_dir else map_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        track_map = load_track_map(map_dir)
        if args.label_modes == "both":
            label_sets = materialize_dual_label_modes(track_map)
        else:
            label_sets = {args.label_modes: materialize_label_mode(track_map, args.label_modes)}

        manifest = {
            "map_id": track_map.map_id,
            "track_name": track_map.track_name,
            "outputs": {},
        }

        for label_mode, records in label_sets.items():
            label_path = output_dir / f"labels_{label_mode}.csv"
            stats_path = output_dir / f"lookahead_stats_{label_mode}.json"
            _write_csv(label_path, label_records_to_rows(records))
            stats = summarize_lookahead_stats(track_map, label_mode, records)
            stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
            manifest["outputs"][label_mode] = {
                "labels_csv": str(label_path),
                "lookahead_stats_json": str(stats_path),
                "record_count": len(records),
            }
            print(f"[labels] mode={label_mode} labels_csv={label_path}")
            print(f"[labels] mode={label_mode} lookahead_stats_json={stats_path}")

        manifest_path = output_dir / "label_artifacts.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[labels] manifest={manifest_path}")
        return

    cfg = dk.load_config(None, args.simulationconfig)
    raw_roots = [Path(args.raw_root).resolve()] if args.raw_root else [Path(part).resolve() for part in str(args.raw_roots).split(",") if part.strip()]
    if not raw_roots:
        raise ValueError("At least one raw dataset root is required.")
    index_root = Path(args.output_dir).resolve() if args.output_dir else raw_roots[0].parent / "index"
    maps_root = Path(args.maps_root).resolve() if args.maps_root else default_maps_root(cfg)
    results = build_phase2_manifests(
        raw_root=raw_roots,
        maps_root=maps_root,
        index_root=index_root,
        target_recovery_ratio=args.target_recovery_ratio,
        rollout_max_share=args.rollout_max_share,
        generated_roads_rollout_max_share=args.generated_roads_rollout_max_share,
    )
    print(f"[manifests] episodes_jsonl={results['episodes_jsonl']}")
    for label_mode, split_paths in sorted(results["sample_manifests"].items()):
        for split, manifest_path in sorted(split_paths.items()):
            print(f"[manifests] mode={label_mode} split={split} jsonl={manifest_path}")
    for report_name, report_path in sorted(results["reports"].items()):
        print(f"[manifests] {report_name}={report_path}")
    print(f"[manifests] manifest={results['manifest_artifacts_json']}")


if __name__ == "__main__":
    main()
