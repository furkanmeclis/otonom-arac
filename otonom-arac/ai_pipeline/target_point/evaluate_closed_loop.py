"""Closed-loop simulator evaluation for target-point pilots."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from target_point.collector import default_maps_root, discover_track_maps
from target_point.controller import TargetPointController
from target_point.pilot import TargetPointPilot
from target_point.sim_session import DonkeySimSession
from target_point.teacher_policy import project_pose_to_centerline
from target_point.track_map import TrackMapArtifact, load_track_map


def _cfg_float(cfg, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _cfg_int(cfg, name: str, default: int) -> int:
    return int(getattr(cfg, name, default))


def _safe_float(value) -> Optional[float]:
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    return cast if math.isfinite(cast) else None


def _p95(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float32), 95))


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _wrap_progress_delta(current_distance: float, previous_distance: float, total_length_m: float) -> float:
    delta = float(current_distance) - float(previous_distance)
    half_length = float(total_length_m) * 0.5
    if delta < -half_length:
        delta += float(total_length_m)
    elif delta > half_length:
        delta -= float(total_length_m)
    return delta


def _track_map_lookup(maps_root: Path) -> Dict[str, Path]:
    if not maps_root.exists():
        return {}
    return discover_track_maps(maps_root)


@dataclass
class EvaluationThresholds:
    offtrack_cte_m: float
    offtrack_centerline_m: float
    offtrack_hold_steps: int
    recovery_trigger_cte_m: float
    recovery_recenter_cte_m: float
    recovery_window_steps: int
    recenter_hold_steps: int
    corner_curvature_threshold: float
    oscillation_steer_threshold: float
    max_progress_delta_m: float


def _thresholds_from_cfg(cfg) -> EvaluationThresholds:
    control_hz = max(1.0, _cfg_float(cfg, "DRIVE_LOOP_HZ", 20.0))
    recovery_window_sec = _cfg_float(cfg, "TARGET_POINT_EVAL_RECOVERY_WINDOW_SEC", 2.0)
    return EvaluationThresholds(
        offtrack_cte_m=_cfg_float(cfg, "TARGET_POINT_EVAL_OFFTRACK_CTE_M", 1.0),
        offtrack_centerline_m=_cfg_float(cfg, "TARGET_POINT_EVAL_OFFTRACK_CENTERLINE_M", 1.1),
        offtrack_hold_steps=max(1, _cfg_int(cfg, "TARGET_POINT_EVAL_OFFTRACK_HOLD_STEPS", 3)),
        recovery_trigger_cte_m=_cfg_float(cfg, "TARGET_POINT_EVAL_RECOVERY_TRIGGER_CTE_M", 0.6),
        recovery_recenter_cte_m=_cfg_float(cfg, "TARGET_POINT_EVAL_RECOVERY_RECENTER_CTE_M", 0.3),
        recovery_window_steps=max(1, int(round(recovery_window_sec * control_hz))),
        recenter_hold_steps=max(1, _cfg_int(cfg, "TARGET_POINT_EVAL_RECENTER_HOLD_STEPS", 3)),
        corner_curvature_threshold=_cfg_float(cfg, "TARGET_POINT_EVAL_CORNER_CURVATURE_THRESHOLD", 0.4),
        oscillation_steer_threshold=_cfg_float(cfg, "TARGET_POINT_EVAL_OSCILLATION_STEER_THRESHOLD", 0.15),
        max_progress_delta_m=_cfg_float(cfg, "TARGET_POINT_EVAL_MAX_PROGRESS_DELTA_M", 3.0),
    )


def _progress_percent(
    track_map: Optional[TrackMapArtifact],
    cumulative_progress_m: float,
    max_map_distance_m: Optional[float],
    start_map_distance_m: Optional[float],
    lap_count: int,
) -> Optional[float]:
    if track_map is None:
        return 100.0 if int(lap_count) > 0 else None

    if track_map.is_closed:
        if int(lap_count) > 0:
            return 100.0
        if track_map.total_length_m <= 1e-6:
            return 0.0
        return float(max(0.0, min(100.0, 100.0 * float(cumulative_progress_m) / float(track_map.total_length_m))))

    if start_map_distance_m is None or max_map_distance_m is None or track_map.total_length_m <= 1e-6:
        return 0.0
    return float(
        max(
            0.0,
            min(
                100.0,
                100.0 * max(0.0, float(max_map_distance_m) - float(start_map_distance_m)) / float(track_map.total_length_m),
            ),
        )
    )


def _failure_context(projected_state, center_error: float, recent_sign_flips: int) -> Dict[str, object]:
    curvature_score = None if projected_state is None else float(projected_state.curvature_score)
    return {
        "center_error_m": float(center_error),
        "curvature_score": curvature_score,
        "in_corner": bool(curvature_score is not None and curvature_score >= 0.4),
        "recent_oscillation_flips": int(recent_sign_flips),
    }


def _corrcoef(values_a: Sequence[float], values_b: Sequence[float]) -> Optional[float]:
    if len(values_a) < 2 or len(values_b) < 2 or len(values_a) != len(values_b):
        return None
    arr_a = np.asarray(values_a, dtype=np.float32)
    arr_b = np.asarray(values_b, dtype=np.float32)
    if float(np.std(arr_a)) <= 1e-6 or float(np.std(arr_b)) <= 1e-6:
        return None
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def evaluate_episode(
    cfg,
    track_name: str,
    split_name: str,
    model_path: Path,
    track_map: Optional[TrackMapArtifact],
    seed: int,
    max_steps: int,
    thresholds: EvaluationThresholds,
) -> Dict[str, object]:
    pilot = TargetPointPilot(cfg)
    pilot.load(model_path.as_posix())
    controller = TargetPointController(cfg)

    steering_history: List[float] = []
    steering_deltas: List[float] = []
    abs_target_x_values: List[float] = []
    abs_steering_values: List[float] = []
    center_errors: List[float] = []
    corner_center_errors: List[float] = []
    corner_speeds: List[float] = []
    corner_abs_steering: List[float] = []
    corner_abs_target_x: List[float] = []
    failure_reason = None
    failure_context = None
    time_to_failure_sec = None
    invalid_prediction_count = 0
    step_count = 0
    offtrack_events = 0
    offtrack_active = False
    offtrack_hold_count = 0
    oscillation_flips = 0
    lap_count = 0
    cumulative_progress_m = 0.0
    max_map_distance_m = None
    start_map_distance_m = None
    previous_map_distance_m = None

    recovery_total = 0
    recovery_success = 0
    recovery_failure = 0
    recovery_event = None

    with DonkeySimSession(cfg, env_name=track_name, seed=seed) as session:
        observation = session.prime()
        recent_flip_window: List[int] = []

        for _ in range(int(max_steps)):
            target_x, target_y = pilot.run(observation.image)
            steer, throttle = controller.run(target_x, target_y)
            abs_target_x_values.append(abs(float(target_x)))
            abs_steering_values.append(abs(float(steer)))
            valid_prediction = _safe_float(target_x) is not None and _safe_float(target_y) is not None and float(target_y) >= float(
                getattr(cfg, "TARGET_POINT_MIN_FORWARD", 0.0)
            )
            if not valid_prediction:
                invalid_prediction_count += 1

            next_observation = session.step(steer, throttle)
            step_count += 1
            lap_count = max(lap_count, int(next_observation.lap_count))

            projected_state = None
            if track_map is not None:
                try:
                    projected_state = project_pose_to_centerline(track_map, *next_observation.pos)
                except Exception:
                    projected_state = None

            centerline_distance = 0.0 if projected_state is None else float(projected_state.distance_to_centerline_m)
            sim_cte = abs(float(next_observation.cte))
            center_error = max(sim_cte, centerline_distance)
            center_errors.append(float(center_error))

            if track_map is not None and projected_state is not None:
                current_distance = float(projected_state.map_distance_m)
                if track_map.is_closed:
                    if previous_map_distance_m is None:
                        previous_map_distance_m = current_distance
                    else:
                        delta = _wrap_progress_delta(current_distance, previous_map_distance_m, track_map.total_length_m)
                        delta = max(-thresholds.max_progress_delta_m, min(thresholds.max_progress_delta_m, delta))
                        if delta > 0.0:
                            cumulative_progress_m += float(delta)
                        previous_map_distance_m = current_distance
                else:
                    if start_map_distance_m is None:
                        start_map_distance_m = current_distance
                        max_map_distance_m = current_distance
                    else:
                        max_map_distance_m = max(float(max_map_distance_m), current_distance)

                if float(projected_state.curvature_score) >= float(thresholds.corner_curvature_threshold):
                    corner_center_errors.append(float(center_error))
                    speed_mps = float(next_observation.forward_vel) if float(next_observation.forward_vel) > 0.0 else float(next_observation.speed)
                    corner_speeds.append(speed_mps)
                    corner_abs_steering.append(abs(float(steer)))
                    corner_abs_target_x.append(abs(float(target_x)))

            if steering_history:
                delta = abs(float(steer) - float(steering_history[-1]))
                steering_deltas.append(delta)
                if (
                    abs(float(steer)) >= float(thresholds.oscillation_steer_threshold)
                    and abs(float(steering_history[-1])) >= float(thresholds.oscillation_steer_threshold)
                    and math.copysign(1.0, float(steer)) != math.copysign(1.0, float(steering_history[-1]))
                ):
                    oscillation_flips += 1
                    recent_flip_window.append(1)
                else:
                    recent_flip_window.append(0)
            steering_history.append(float(steer))
            if len(recent_flip_window) > 20:
                recent_flip_window = recent_flip_window[-20:]

            severe_offtrack_condition = (
                (centerline_distance >= float(thresholds.offtrack_centerline_m) if projected_state is not None else False)
                or sim_cte >= float(thresholds.offtrack_cte_m)
            )
            if str(next_observation.hit).lower() != "none":
                severe_offtrack_condition = True

            if severe_offtrack_condition:
                offtrack_hold_count += 1
            else:
                offtrack_hold_count = 0
                offtrack_active = False

            offtrack_now = str(next_observation.hit).lower() != "none" or offtrack_hold_count >= int(thresholds.offtrack_hold_steps)
            if offtrack_now and not offtrack_active:
                offtrack_events += 1
                offtrack_active = True

            if recovery_event is None and center_error >= float(thresholds.recovery_trigger_cte_m):
                recovery_event = {
                    "start_step": int(step_count),
                    "centered_steps": 0,
                }
                recovery_total += 1

            if recovery_event is not None:
                if center_error <= float(thresholds.recovery_recenter_cte_m):
                    recovery_event["centered_steps"] = int(recovery_event["centered_steps"]) + 1
                else:
                    recovery_event["centered_steps"] = 0

                window_age = int(step_count) - int(recovery_event["start_step"])
                if int(recovery_event["centered_steps"]) >= int(thresholds.recenter_hold_steps):
                    recovery_success += 1
                    recovery_event = None
                elif offtrack_now or str(next_observation.hit).lower() != "none":
                    recovery_failure += 1
                    recovery_event = None
                elif window_age >= int(thresholds.recovery_window_steps):
                    recovery_failure += 1
                    recovery_event = None

            completion_percent = _progress_percent(
                track_map=track_map,
                cumulative_progress_m=cumulative_progress_m,
                max_map_distance_m=max_map_distance_m,
                start_map_distance_m=start_map_distance_m,
                lap_count=lap_count,
            )

            lap_complete = bool(lap_count > 0) or (completion_percent is not None and float(completion_percent) >= 99.0)
            if failure_reason is None and offtrack_now:
                failure_reason = "collision" if str(next_observation.hit).lower() != "none" else "offtrack"
                time_to_failure_sec = float(session.elapsed_seconds)
                failure_context = _failure_context(projected_state, center_error, sum(recent_flip_window))
            elif failure_reason is None and next_observation.done and not lap_complete:
                failure_reason = "episode_done"
                time_to_failure_sec = float(session.elapsed_seconds)
                failure_context = _failure_context(projected_state, center_error, sum(recent_flip_window))

            observation = next_observation
            if failure_reason is not None or next_observation.done:
                break

        if recovery_event is not None:
            recovery_failure += 1

        duration_sec = float(session.elapsed_seconds)
        if time_to_failure_sec is None:
            time_to_failure_sec = duration_sec

    if track_map is None:
        completion_percent = 100.0 if lap_count > 0 else None
    else:
        completion_percent = _progress_percent(
            track_map=track_map,
            cumulative_progress_m=cumulative_progress_m,
            max_map_distance_m=max_map_distance_m,
            start_map_distance_m=start_map_distance_m,
            lap_count=lap_count,
        )

    sim_minutes = duration_sec / 60.0 if duration_sec > 1e-6 else 0.0
    offtrack_frequency = None if sim_minutes <= 0.0 else float(offtrack_events / sim_minutes)
    oscillation_rate = None if duration_sec <= 0.0 else float(oscillation_flips / duration_sec)

    return {
        "track_name": track_name,
        "split_name": split_name,
        "seed": int(seed),
        "map_available": bool(track_map is not None),
        "map_id": None if track_map is None else track_map.map_id,
        "map_is_closed": None if track_map is None else bool(track_map.is_closed),
        "steps": int(step_count),
        "duration_sec": float(duration_sec),
        "lap_count": int(lap_count),
        "completion_percent": completion_percent,
        "completion_method": "map_progress" if track_map is not None else "lap_count_only",
        "offtrack_events": int(offtrack_events),
        "offtrack_frequency_per_min": offtrack_frequency,
        "time_to_failure_sec": float(time_to_failure_sec),
        "recovery_event_count": int(recovery_total),
        "recovery_success_count": int(recovery_success),
        "recovery_failure_count": int(recovery_failure),
        "recovery_success_rate": None if recovery_total <= 0 else float(recovery_success / recovery_total),
        "mean_abs_cte_m": _mean(center_errors),
        "p95_abs_cte_m": _p95(center_errors),
        "mean_abs_target_x": _mean(abs_target_x_values),
        "p95_abs_target_x": _p95(abs_target_x_values),
        "mean_abs_steering": _mean(abs_steering_values),
        "p95_abs_steering": _p95(abs_steering_values),
        "abs_steering_vs_target_x_corr": _corrcoef(abs_target_x_values, abs_steering_values),
        "steering_smoothness_p95": _p95(steering_deltas),
        "oscillation_flips": int(oscillation_flips),
        "oscillation_rate_hz": oscillation_rate,
        "corner_frame_count": int(len(corner_center_errors)),
        "corner_mean_abs_cte_m": _mean(corner_center_errors),
        "corner_p95_abs_cte_m": _p95(corner_center_errors),
        "corner_mean_speed_mps": _mean(corner_speeds),
        "corner_mean_abs_target_x": _mean(corner_abs_target_x),
        "corner_mean_abs_steering": _mean(corner_abs_steering),
        "invalid_prediction_rate": 0.0 if step_count <= 0 else float(invalid_prediction_count / step_count),
        "failure_reason": failure_reason,
        "failure_context": failure_context,
    }


def _aggregate_track_results(track_name: str, split_name: str, episode_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    durations = [float(item["duration_sec"]) for item in episode_results]
    total_minutes = sum(durations) / 60.0 if durations else 0.0
    total_recovery_events = sum(int(item["recovery_event_count"]) for item in episode_results)
    total_recovery_success = sum(int(item["recovery_success_count"]) for item in episode_results)
    completion_values = [float(item["completion_percent"]) for item in episode_results if item.get("completion_percent") is not None]
    offtrack_events = sum(int(item["offtrack_events"]) for item in episode_results)
    invalid_rates = [float(item["invalid_prediction_rate"]) for item in episode_results]
    mean_cte_values = [float(item["mean_abs_cte_m"]) for item in episode_results if item.get("mean_abs_cte_m") is not None]
    p95_cte_values = [float(item["p95_abs_cte_m"]) for item in episode_results if item.get("p95_abs_cte_m") is not None]
    mean_abs_target_x_values = [float(item["mean_abs_target_x"]) for item in episode_results if item.get("mean_abs_target_x") is not None]
    mean_abs_steering_values = [float(item["mean_abs_steering"]) for item in episode_results if item.get("mean_abs_steering") is not None]
    steering_target_corrs = [
        float(item["abs_steering_vs_target_x_corr"])
        for item in episode_results
        if item.get("abs_steering_vs_target_x_corr") is not None
    ]
    smoothness_values = [float(item["steering_smoothness_p95"]) for item in episode_results if item.get("steering_smoothness_p95") is not None]
    oscillation_rates = [float(item["oscillation_rate_hz"]) for item in episode_results if item.get("oscillation_rate_hz") is not None]
    corner_mean_values = [float(item["corner_mean_abs_cte_m"]) for item in episode_results if item.get("corner_mean_abs_cte_m") is not None]
    corner_p95_values = [float(item["corner_p95_abs_cte_m"]) for item in episode_results if item.get("corner_p95_abs_cte_m") is not None]
    corner_target_x_values = [float(item["corner_mean_abs_target_x"]) for item in episode_results if item.get("corner_mean_abs_target_x") is not None]
    corner_steering_values = [float(item["corner_mean_abs_steering"]) for item in episode_results if item.get("corner_mean_abs_steering") is not None]
    failure_counter = Counter(str(item["failure_reason"]) for item in episode_results if item.get("failure_reason"))
    map_available = all(bool(item["map_available"]) for item in episode_results)

    return {
        "track_name": track_name,
        "split_name": split_name,
        "episode_count": len(episode_results),
        "map_available": bool(map_available),
        "completion_percent_mean": _mean(completion_values),
        "completion_percent_min": None if not completion_values else float(min(completion_values)),
        "offtrack_events": int(offtrack_events),
        "offtrack_frequency_per_min": None if total_minutes <= 0.0 else float(offtrack_events / total_minutes),
        "time_to_failure_sec_mean": _mean([float(item["time_to_failure_sec"]) for item in episode_results]),
        "time_to_failure_sec_min": float(min(float(item["time_to_failure_sec"]) for item in episode_results)),
        "recovery_success_rate": None if total_recovery_events <= 0 else float(total_recovery_success / total_recovery_events),
        "recovery_event_count": int(total_recovery_events),
        "mean_abs_cte_m": _mean(mean_cte_values),
        "p95_abs_cte_m": _mean(p95_cte_values),
        "mean_abs_target_x": _mean(mean_abs_target_x_values),
        "mean_abs_steering": _mean(mean_abs_steering_values),
        "abs_steering_vs_target_x_corr": _mean(steering_target_corrs),
        "steering_smoothness_p95": _mean(smoothness_values),
        "oscillation_rate_hz": _mean(oscillation_rates),
        "corner_mean_abs_cte_m": _mean(corner_mean_values),
        "corner_p95_abs_cte_m": _mean(corner_p95_values),
        "corner_mean_abs_target_x": _mean(corner_target_x_values),
        "corner_mean_abs_steering": _mean(corner_steering_values),
        "invalid_prediction_rate": _mean(invalid_rates),
        "failure_reasons": dict(sorted(failure_counter.items())),
    }


def evaluate_closed_loop(
    cfg,
    model_path: str | Path,
    tracks: Optional[Sequence[str]] = None,
    maps_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    episodes_per_track: Optional[int] = None,
    max_steps: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, object]:
    model_path = Path(model_path).resolve()
    maps_root_path = Path(maps_root).resolve() if maps_root else default_maps_root(cfg)
    episodes_per_track = int(episodes_per_track or getattr(cfg, "TARGET_POINT_EVAL_EPISODES_PER_TRACK", 1))
    max_steps = int(max_steps or getattr(cfg, "TARGET_POINT_EVAL_MAX_STEPS", 2200))
    thresholds = _thresholds_from_cfg(cfg)

    requested_tracks = list(tracks) if tracks else list(getattr(cfg, "TARGET_POINT_EVAL_TRACKS", ()))
    if not requested_tracks:
        raise ValueError("No evaluation tracks provided. Pass --tracks or set TARGET_POINT_EVAL_TRACKS.")

    track_map_dirs = _track_map_lookup(maps_root_path)
    loaded_maps: Dict[str, TrackMapArtifact] = {}
    for track_name, map_dir in track_map_dirs.items():
        loaded_maps[track_name] = load_track_map(map_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_root = (
        Path(output_dir).resolve()
        if output_dir
        else Path(getattr(cfg, "TARGET_POINT_REPORTS_PATH")).resolve() / f"closed_loop_{timestamp}"
    )
    report_root.mkdir(parents=True, exist_ok=True)

    episode_results: List[Dict[str, object]] = []
    tracks_summary: List[Dict[str, object]] = []

    split_lookup = {}
    for split_name, split_tracks in dict(getattr(cfg, "TARGET_POINT_TRACK_SPLITS", {})).items():
        for track_name in split_tracks:
            split_lookup[str(track_name)] = str(split_name)

    for track_name in requested_tracks:
        split_name = split_lookup.get(track_name, "unknown")
        track_map = loaded_maps.get(track_name)
        per_track_results = []
        for episode_index in range(int(episodes_per_track)):
            episode_seed = int(seed) + episode_index
            result = evaluate_episode(
                cfg=cfg,
                track_name=track_name,
                split_name=split_name,
                model_path=model_path,
                track_map=track_map,
                seed=episode_seed,
                max_steps=max_steps,
                thresholds=thresholds,
            )
            result["episode_index"] = int(episode_index)
            per_track_results.append(result)
            episode_results.append(result)

        tracks_summary.append(_aggregate_track_results(track_name, split_name, per_track_results))

    summary = {
        "timestamp_utc": timestamp,
        "model_path": model_path.as_posix(),
        "maps_root": maps_root_path.as_posix(),
        "thresholds": {
            "offtrack_cte_m": thresholds.offtrack_cte_m,
            "offtrack_centerline_m": thresholds.offtrack_centerline_m,
            "offtrack_hold_steps": thresholds.offtrack_hold_steps,
            "recovery_trigger_cte_m": thresholds.recovery_trigger_cte_m,
            "recovery_recenter_cte_m": thresholds.recovery_recenter_cte_m,
            "recovery_window_steps": thresholds.recovery_window_steps,
            "recenter_hold_steps": thresholds.recenter_hold_steps,
            "corner_curvature_threshold": thresholds.corner_curvature_threshold,
            "oscillation_steer_threshold": thresholds.oscillation_steer_threshold,
            "max_progress_delta_m": thresholds.max_progress_delta_m,
        },
        "tracks": tracks_summary,
        "episodes": episode_results,
    }

    summary_path = report_root / "closed_loop_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    episode_jsonl = report_root / "closed_loop_episodes.jsonl"
    with episode_jsonl.open("w", encoding="utf-8") as stream:
        for row in episode_results:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "report_root": report_root.as_posix(),
        "summary_json": summary_path.as_posix(),
        "episodes_jsonl": episode_jsonl.as_posix(),
        "tracks": tracks_summary,
        "episodes": episode_results,
    }
