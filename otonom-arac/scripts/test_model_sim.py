#!/usr/bin/env python3
"""Test trained model in simulator."""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ai_pipeline"))

from target_point.pilot import TargetPointPilot
from target_point.sim_session import SimulatorSession
from target_point.mapping import discover_track_maps

# Config
MODEL_PATH = "models/sim_multitrack_v1.keras"
TRACK = "donkey-generated-roads-v0"
LAPS = 2
MAX_STEPS = 2000

def main():
    print(f"[test] Loading model from {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"[error] Model not found: {MODEL_PATH}")
        return 1

    # Load config
    import config as cfg

    # Create pilot
    pilot = TargetPointPilot(cfg)
    pilot.load(MODEL_PATH)
    print("[pilot] Model loaded")

    # Discover track maps
    print(f"[test] Discovering track maps...")
    track_maps = discover_track_maps("data/sim_unified_maps")
    if not track_maps:
        print("[error] No track maps found")
        return 1
    print(f"[test] Found {len(track_maps)} tracks: {list(track_maps.keys())}")

    if TRACK not in track_maps:
        print(f"[error] Track '{TRACK}' not found. Available: {list(track_maps.keys())}")
        return 1

    track_map = track_maps[TRACK]
    print(f"[test] Using track: {TRACK}")
    print(f"[test] Map: {track_map}")

    # Run simulation
    print(f"\n[test] Starting simulation: {LAPS} laps, max {MAX_STEPS} steps/lap")

    session = SimulatorSession(cfg)
    try:
        results = session.run_closed_loop(
            track=TRACK,
            pilot=pilot,
            track_map=track_map,
            laps=LAPS,
            max_steps_per_lap=MAX_STEPS,
        )

        print(f"\n[results] Simulation completed:")
        print(f"  Laps completed: {results['laps_completed']}")
        print(f"  Total steps: {results['total_steps']}")
        print(f"  Avg speed: {results.get('avg_speed', 'N/A')}")
        print(f"  Success rate: {results.get('success_rate', 'N/A')}%")

        return 0
    except Exception as e:
        print(f"[error] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        pilot.shutdown()

if __name__ == "__main__":
    sys.exit(main())
