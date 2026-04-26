#!/usr/bin/env python3
"""Run closed-loop simulator evaluation for a target-point model."""

from __future__ import annotations

import argparse
from pathlib import Path

import donkeycar as dk

from target_point.evaluate_closed_loop import evaluate_closed_loop


def _parse_track_list(values):
    tracks = []
    for value in values or ():
        for chunk in str(value).split(","):
            track = chunk.strip()
            if track:
                tracks.append(track)
    return tracks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained .keras target-point model, e.g. models/target_point_phase4_adaptive_generalize.keras",
    )
    parser.add_argument("--simulationconfig", default="simulationconfig.py", help="Config file to load. [default: simulationconfig.py]")
    parser.add_argument("--maps-root", default=None, help="Optional Phase 1 maps root. Defaults to accepted maps under artifacts.")
    parser.add_argument("--output-dir", default=None, help="Optional report output directory.")
    parser.add_argument("--tracks", action="append", help="Comma-separated simulator track ids to evaluate.")
    parser.add_argument("--episodes-per-track", type=int, default=None, help="Optional episode count override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max-step override.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for deterministic episodes. [default: from simulationconfig]")
    parser.add_argument("--base-throttle", type=float, default=None, help="Max throttle on straights. [default: from simulationconfig]")
    parser.add_argument("--min-throttle", type=float, default=None, help="Min throttle on sharp curves. [default: from simulationconfig]")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = dk.load_config(None, args.simulationconfig)
    if args.base_throttle is not None:
        cfg.TARGET_POINT_BASE_THROTTLE = args.base_throttle
    if args.min_throttle is not None:
        cfg.TARGET_POINT_MIN_THROTTLE = args.min_throttle
    results = evaluate_closed_loop(
        cfg=cfg,
        model_path=Path(args.model).resolve(),
        tracks=_parse_track_list(args.tracks),
        maps_root=args.maps_root,
        output_dir=args.output_dir,
        episodes_per_track=args.episodes_per_track,
        max_steps=args.max_steps,
        seed=args.seed if args.seed is not None else 42,
    )

    print(f"[closed_loop] report_root={results['report_root']}")
    print(f"[closed_loop] summary_json={results['summary_json']}")
    print(f"[closed_loop] episodes_jsonl={results['episodes_jsonl']}")
    for track_summary in results["tracks"]:
        completion = track_summary["completion_percent_mean"]
        completion_text = "n/a" if completion is None else f"{completion:.1f}"
        recovery = track_summary["recovery_success_rate"]
        recovery_text = "n/a" if recovery is None else f"{100.0 * recovery:.1f}%"
        offtrack = track_summary["offtrack_frequency_per_min"]
        offtrack_text = "n/a" if offtrack is None else f"{offtrack:.2f}/min"
        ttf = track_summary["time_to_failure_sec_mean"]
        ttf_text = "n/a" if ttf is None else f"{ttf:.2f}s"
        print(
            "[closed_loop] "
            f"track={track_summary['track_name']} "
            f"split={track_summary['split_name']} "
            f"completion={completion_text}% "
            f"offtrack={offtrack_text} "
            f"ttf={ttf_text} "
            f"recovery={recovery_text} "
            f"failure_reasons={track_summary['failure_reasons']}"
        )


if __name__ == "__main__":
    main()

