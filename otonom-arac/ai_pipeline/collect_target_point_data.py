#!/usr/bin/env python3
"""Phase 1 mapping plus Phase 2/3 dataset collection entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import donkeycar as dk

from target_point.collector import (
    COLLECTION_PROFILE_PHASE55,
    collect_dataset,
    collect_rollout_dataset,
    default_maps_root,
    parse_track_list,
    track_split_plan,
)
from target_point.mapping import build_phase1_map
from target_point.teacher_policy import COLLECTION_PROFILE_PHASE2, COLLECTION_PROFILE_PHASE3
from target_point.track_map import MAP_SPACING_METERS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="map",
        choices=("map", "collect", "rollout_collect"),
        help="Run Phase 1 mapping or Phase 2 baseline collection. [default: map]",
    )
    parser.add_argument("--simulationconfig", default="simulationconfig.py", help="Config file to load. [default: simulationconfig.py]")
    parser.add_argument("--track", default=None, help="Gym track id for --task map, e.g. donkey-generated-roads-v0")
    parser.add_argument("--laps", type=int, default=2, help="Number of clean laps to map. [default: 2]")
    parser.add_argument("--max-steps", type=int, default=4000, help="Maximum simulator steps before aborting. [default: 4000]")
    parser.add_argument("--spacing-m", type=float, default=MAP_SPACING_METERS, help="Centerline waypoint spacing in meters. [default: 0.25]")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for scripted runs. [default: 42]")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root. Defaults to <ARTIFACTS_PATH>/maps for mapping and <DATA_PATH>/target_point_phase2 for collection.",
    )
    parser.add_argument("--maps-root", default=None, help="Phase 1 maps root for --task collect.")
    parser.add_argument("--train-tracks", action="append", help="Comma-separated train tracks for --task collect.")
    parser.add_argument("--val-tracks", action="append", help="Comma-separated val tracks for --task collect.")
    parser.add_argument("--episodes-per-track", type=int, default=1, help="Episodes per track for --task collect. [default: 1]")
    parser.add_argument(
        "--min-samples-per-track",
        type=int,
        default=None,
        help="Optional minimum raw sample budget per split/track for --task collect; keeps resetting open tracks until reached.",
    )
    parser.add_argument(
        "--collection-profile",
        default=COLLECTION_PROFILE_PHASE2,
        choices=(COLLECTION_PROFILE_PHASE2, COLLECTION_PROFILE_PHASE3),
        help="Collection teacher profile for --task collect. [default: phase2_low_noise]",
    )
    parser.add_argument(
        "--nominal-only-tracks",
        action="append",
        help="Comma-separated tracks that should stay nominal-only during collection.",
    )
    parser.add_argument(
        "--fixed-throttle",
        type=float,
        default=None,
        help="Optional fixed teacher throttle for --task collect, e.g. 0.10.",
    )
    parser.add_argument(
        "--remap-generated-roads",
        action="store_true",
        help="For donkey-generated-roads-v0, build a fresh Phase 1 map for each episode seed before collection.",
    )
    parser.add_argument("--driver-model", default=None, help="Closed-loop model path for --task rollout_collect.")
    parser.add_argument(
        "--min-usable-recovery-samples",
        type=int,
        default=None,
        help="Target usable recovery/failure-margin samples per train track for --task rollout_collect.",
    )
    return parser.parse_args()
def run_mapping(args: argparse.Namespace, cfg) -> None:
    if not args.track:
        raise ValueError("--track is required when --task map")

    output_root = Path(args.output_root).resolve() if args.output_root else Path(cfg.ARTIFACTS_PATH).resolve() / "maps"
    build_phase1_map(
        cfg=cfg,
        track_name=args.track,
        output_root=output_root,
        seed=int(args.seed),
        laps=int(args.laps),
        max_steps=int(args.max_steps),
        spacing_m=float(args.spacing_m),
    )


def run_collection(args: argparse.Namespace, cfg) -> None:
    if args.output_root:
        dataset_root = Path(args.output_root).resolve()
    else:
        default_dataset_name = "target_point_phase3" if args.collection_profile == COLLECTION_PROFILE_PHASE3 else "target_point_phase2"
        dataset_root = Path(cfg.DATA_PATH).resolve() / default_dataset_name
    maps_root = Path(args.maps_root).resolve() if args.maps_root else default_maps_root(cfg)
    nominal_only_tracks = parse_track_list(args.nominal_only_tracks)
    if args.collection_profile == COLLECTION_PROFILE_PHASE3 and not nominal_only_tracks:
        nominal_only_tracks = ["donkey-generated-roads-v0"]
    split_tracks = track_split_plan(
        cfg,
        train_tracks=parse_track_list(args.train_tracks),
        val_tracks=parse_track_list(args.val_tracks),
    )
    results = collect_dataset(
        cfg=cfg,
        maps_root=maps_root,
        dataset_root=dataset_root,
        split_tracks=split_tracks,
        episodes_per_track=int(args.episodes_per_track),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        collection_profile=str(args.collection_profile),
        nominal_only_tracks=nominal_only_tracks,
        fixed_throttle=args.fixed_throttle,
        min_samples_per_track=args.min_samples_per_track,
        remap_generated_roads=bool(args.remap_generated_roads),
    )
    print(f"[collect] dataset_root={results['dataset_root']}")
    print(f"[collect] raw_index_path={results['raw_index_path']}")
    print(f"[collect] episode_count={results['episode_count']}")


def run_rollout_collection(args: argparse.Namespace, cfg) -> None:
    if args.output_root:
        dataset_root = Path(args.output_root).resolve()
    else:
        dataset_root = Path(cfg.DATA_PATH).resolve() / "target_point_phase55_bootstrap"
    maps_root = Path(args.maps_root).resolve() if args.maps_root else default_maps_root(cfg)
    nominal_only_tracks = parse_track_list(args.nominal_only_tracks)
    if not nominal_only_tracks:
        nominal_only_tracks = ["donkey-generated-roads-v0"]
    train_tracks = parse_track_list(args.train_tracks) or list(getattr(cfg, "TARGET_POINT_TRAIN_TRACKS", ()))
    results = collect_rollout_dataset(
        cfg=cfg,
        maps_root=maps_root,
        dataset_root=dataset_root,
        train_tracks=train_tracks,
        max_episodes_per_track=int(args.episodes_per_track),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        driver_model_path=args.driver_model,
        label_mode=str(getattr(cfg, "TARGET_POINT_LABEL_MODE", "adaptive_v1")),
        nominal_only_tracks=nominal_only_tracks,
        min_usable_recovery_samples=args.min_usable_recovery_samples,
    )
    print(f"[rollout] dataset_root={results['dataset_root']}")
    print(f"[rollout] raw_index_path={results['raw_index_path']}")
    print(f"[rollout] episode_count={results['episode_count']}")
    print(f"[rollout] driver_model_path={results['driver_model_path']}")
    if results.get("driver_report_path"):
        print(f"[rollout] driver_report_path={results['driver_report_path']}")
    print(f"[rollout] summary_path={results['summary_path']}")


def main() -> None:
    args = _parse_args()
    cfg = dk.load_config(None, args.simulationconfig)
    if args.task == "map":
        run_mapping(args, cfg)
    elif args.task == "collect":
        run_collection(args, cfg)
    else:
        run_rollout_collection(args, cfg)


if __name__ == "__main__":
    main()
