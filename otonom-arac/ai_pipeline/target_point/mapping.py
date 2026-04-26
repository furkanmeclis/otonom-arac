"""Shared Phase 1 mapping helpers for track-map generation."""

from __future__ import annotations

import math
from pathlib import Path

from target_point.sim_session import DonkeySimSession
from target_point.teacher_policy import CleanMappingTeacher
from target_point.track_map import MAP_SPACING_METERS, MapTracePoint, build_track_map, save_track_map


def _wrap_delta_deg(current: float, previous: float) -> float:
    return ((float(current) - float(previous) + 180.0) % 360.0) - 180.0


def _trace_point_from_observation(observation, action, step_index: int, elapsed_sec: float, loop_index: int) -> MapTracePoint:
    return MapTracePoint(
        step_index=int(step_index),
        elapsed_sec=float(elapsed_sec),
        loop_index=int(loop_index),
        lap_count=int(observation.lap_count),
        pos_x=float(observation.pos[0]),
        pos_y=float(observation.pos[1]),
        pos_z=float(observation.pos[2]),
        yaw_deg=float(observation.yaw_deg),
        pitch_deg=float(observation.pitch_deg),
        roll_deg=float(observation.roll_deg),
        cte=float(observation.cte),
        speed=float(observation.speed),
        forward_vel=float(observation.forward_vel),
        steering=float(action.steering),
        throttle=float(action.throttle),
        brake=float(action.brake),
        hit=str(observation.hit),
    )


def build_phase1_map(
    cfg,
    track_name: str,
    output_root: str | Path,
    seed: int,
    laps: int = 1,
    max_steps: int = 4000,
    spacing_m: float = MAP_SPACING_METERS,
    open_segment_target_distance_m: float = 80.0,
) -> dict[str, object]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    teacher = CleanMappingTeacher(
        control_hz=float(getattr(cfg, "DRIVE_LOOP_HZ", 20.0)),
        max_safe_cte=10.0,
        max_steer_delta=0.15,
        steer_smoothing=0.50,
        cte_gain=1.80,
    )
    trace = []
    detected_loops = 0
    global_elapsed_sec = 0.0
    dt = 1.0 / max(float(getattr(cfg, "DRIVE_LOOP_HZ", 20.0)), 1.0)
    used_open_fallback = False
    allow_open_fallback = str(track_name) == "donkey-generated-roads-v0"
    target_distance = float(open_segment_target_distance_m) if allow_open_fallback else None

    with DonkeySimSession(cfg, env_name=track_name, seed=seed) as session:
        correction_sign = teacher.infer_cte_correction_sign(session)
        print(f"[map] track={track_name} inferred_cte_correction_sign={correction_sign}")

        while detected_loops < int(laps):
            teacher.reset()
            observation = session.prime()
            start_x, _, start_z = observation.pos
            start_yaw = float(observation.yaw_deg)
            previous_x, previous_z = float(start_x), float(start_z)
            episode_travel_distance = 0.0
            segment_completed = False

            for episode_step in range(int(max_steps)):
                action = teacher.compute_action(observation)
                trace.append(
                    _trace_point_from_observation(
                        observation=observation,
                        action=action,
                        step_index=len(trace),
                        elapsed_sec=global_elapsed_sec,
                        loop_index=detected_loops,
                    )
                )
                global_elapsed_sec += dt

                current_x, _, current_z = observation.pos
                episode_travel_distance += math.hypot(float(current_x) - previous_x, float(current_z) - previous_z)
                previous_x, previous_z = float(current_x), float(current_z)

                if target_distance is not None and episode_step >= 200 and episode_travel_distance >= target_distance:
                    used_open_fallback = True
                    detected_loops += 1
                    segment_completed = True
                    print(
                        f"[map] open_segment_distance={detected_loops} "
                        f"episode_step={episode_step} travel_distance_m={episode_travel_distance:.2f}"
                    )
                    break

                close_to_start = math.hypot(float(current_x) - float(start_x), float(current_z) - float(start_z)) <= 1.5
                heading_aligned = abs(_wrap_delta_deg(float(observation.yaw_deg), start_yaw)) <= 35.0
                enough_progress = episode_step >= 120 and episode_travel_distance >= 20.0
                if close_to_start and heading_aligned and enough_progress:
                    detected_loops += 1
                    segment_completed = True
                    print(
                        f"[map] detected_loop={detected_loops} "
                        f"episode_step={episode_step} travel_distance_m={episode_travel_distance:.2f} "
                        f"sim_lap_count={observation.lap_count}"
                    )
                    break

                if observation.hit != "none":
                    raise RuntimeError(f"Mapping aborted due to simulator collision: hit={observation.hit}")
                if observation.done:
                    print(
                        f"[map] sim_done_signal step={episode_step} travel={episode_travel_distance:.2f}m "
                        f"cte={observation.cte:.3f} pos=({observation.pos[0]:.1f},{observation.pos[2]:.1f})"
                    )
                    if episode_step < 80:
                        # Ignore early done signals (start line crossing)
                        observation = session.step(action.steering, action.throttle, action.brake)
                        continue
                    if episode_travel_distance >= 20.0 and abs(float(observation.cte)) <= 1.5:
                        detected_loops += 1
                        segment_completed = True
                        print(
                            f"[map] simulator_closed_segment={detected_loops} "
                            f"episode_step={episode_step} travel_distance_m={episode_travel_distance:.2f} "
                            f"cte={observation.cte:.3f}"
                        )
                        break
                    raise RuntimeError(
                        f"Mapping aborted: done at step={episode_step} travel={episode_travel_distance:.2f}m cte={observation.cte:.3f}"
                    )

                observation = session.step(action.steering, action.throttle, action.brake)
            else:
                if allow_open_fallback and episode_travel_distance >= 40.0:
                    used_open_fallback = True
                    detected_loops += 1
                    segment_completed = True
                    print(
                        f"[map] open_segment={detected_loops} "
                        f"episode_step={max_steps} travel_distance_m={episode_travel_distance:.2f}"
                    )
                else:
                    raise RuntimeError(
                        f"Mapping aborted after reaching max steps ({max_steps}) before finishing segment {detected_loops + 1}."
                    )

            if not segment_completed:
                raise RuntimeError("Mapping ended without recording a usable segment.")

    track_map = build_track_map(
        trace,
        track_name=track_name,
        spacing_m=float(spacing_m),
        closed=not used_open_fallback,
    )
    output_dir = save_track_map(
        track_map,
        output_root=output_root,
        extra_metadata={
            "seed": int(seed),
            "laps_requested": int(laps),
            "detected_loops": int(detected_loops),
            "used_open_fallback": bool(used_open_fallback),
            "controller": teacher.summary(),
        },
    )

    print(f"[map] map_id={track_map.map_id}")
    print(f"[map] output_dir={output_dir}")
    print(f"[map] total_length_m={track_map.total_length_m:.3f}")
    print(f"[map] centerline_waypoints={len(track_map.centerline)}")
    print(f"[map] raw_trace_points={len(track_map.raw_trace)}")
    return {
        "map_id": track_map.map_id,
        "output_dir": output_dir,
        "track_map": track_map,
        "total_length_m": float(track_map.total_length_m),
        "centerline_waypoints": int(len(track_map.centerline)),
        "raw_trace_points": int(len(track_map.raw_trace)),
        "used_open_fallback": bool(used_open_fallback),
    }
