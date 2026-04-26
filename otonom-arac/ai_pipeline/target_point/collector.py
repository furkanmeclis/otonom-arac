"""Phase 2 and Phase 3 dataset collection on top of Phase 1 map artifacts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from target_point.controller import TargetPointController
from target_point.domain_randomization import DomainProfile, apply_domain_profile, sample_domain_profile
from target_point.mapping import build_phase1_map
from target_point.sim_session import DonkeySimSession
from target_point.teacher_policy import (
    ADAPTIVE_V1,
    CleanMappingTeacher,
    LowNoiseTargetPointTeacher,
    LowNoiseTeacherOutput,
    materialize_label_from_state,
    project_pose_to_centerline,
)
from target_point.track_map import MAP_SPACING_METERS, MapTracePoint, TrackMapArtifact, build_track_map, load_track_map, save_track_map

COLLECTION_PROFILE_PHASE55 = "phase55_model_rollout"


@dataclass(frozen=True)
class RawEpisodeSample:
    sample_id: str
    episode_id: str
    frame_index: int
    elapsed_sec: float
    split: str
    track_name: str
    map_id: str
    is_closed_map: bool
    image_rel_path: str
    scenario: str
    disturbance_type: str
    collection_profile: str
    domain_profile_id: str
    domain_road_appearance: str
    domain_edge_appearance: str
    domain_environment_appearance: str
    domain_brightness: float
    domain_contrast: float
    domain_saturation: float
    domain_rgb_shift_r: int
    domain_rgb_shift_g: int
    domain_rgb_shift_b: int
    domain_blur_radius: float
    domain_jpeg_quality: int
    pos_x: float
    pos_y: float
    pos_z: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    cte: float
    speed: float
    forward_vel: float
    lap_count: int
    last_lap_time: float
    hit: str
    done: bool
    map_distance_m: float
    waypoint_index: int
    distance_to_centerline_m: float
    centerline_pos_x: float
    centerline_pos_y: float
    centerline_pos_z: float
    centerline_yaw_deg: float
    curvature_radpm: float
    delta_heading_2m_deg: float
    curvature_score: float
    reference_speed_mps: float
    teacher_label_mode: str
    clean_lookahead_m: float
    clean_target_x: float
    clean_target_y: float
    noise_target_x: float
    noise_target_y: float
    lane_bias_offset_x: float
    steering_noise: float
    throttle_noise: float
    teacher_commanded_steering: float
    teacher_commanded_throttle: float
    steering_lag_frames: int
    safety_override_active: bool
    recovery_steering_bias: float
    recovery_target_offset_x: float
    recovery_throttle_scale: float
    applied_lookahead_m: float
    applied_target_x: float
    applied_target_y: float
    teacher_steering: float
    teacher_throttle: float
    teacher_brake: float
    tail_clamped: bool
    driver_source: str = "teacher"
    driver_model_path: str = ""
    model_target_x: float = float("nan")
    model_target_y: float = float("nan")
    model_steering: float = float("nan")
    model_throttle: float = float("nan")
    invalid_prediction: bool = False
    rollout_event: str = "none"
    rollout_event_id: int = -1
    deviation_active: bool = False
    failure_margin: bool = False
    usable_recovery_sample: bool = False


@dataclass(frozen=True)
class RolloutCollectionThresholds:
    first_deviation_cte_m: float
    first_deviation_centerline_m: float
    recenter_cte_m: float
    recenter_centerline_m: float
    recenter_hold_steps: int
    failure_margin_steps: int

    def is_first_deviation(self, cte_m: float, centerline_m: float) -> bool:
        return abs(float(cte_m)) >= self.first_deviation_cte_m or float(centerline_m) >= self.first_deviation_centerline_m

    def is_recentered(self, cte_m: float, centerline_m: float) -> bool:
        return abs(float(cte_m)) <= self.recenter_cte_m and float(centerline_m) <= self.recenter_centerline_m


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value)).strip("-")


def default_maps_root(cfg) -> Path:
    artifacts_root = Path(getattr(cfg, "ARTIFACTS_PATH")).resolve()
    acceptance_root = artifacts_root / "maps_phase1_acceptance"
    if acceptance_root.exists():
        return acceptance_root
    return artifacts_root / "maps"


def default_reports_root(cfg) -> Path:
    return Path(getattr(cfg, "TARGET_POINT_REPORTS_PATH", Path(getattr(cfg, "ARTIFACTS_PATH")).resolve() / "target_point" / "reports")).resolve()


def _mean_metric(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return float("-inf")
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def select_best_rollout_driver_model(cfg, reports_root: str | Path | None = None) -> Tuple[Path, Path, Dict[str, float]]:
    reports_root = Path(reports_root).resolve() if reports_root else default_reports_root(cfg)
    best_model_path: Optional[Path] = None
    best_report_path: Optional[Path] = None
    best_score: Optional[Tuple[float, float, float]] = None
    best_summary: Dict[str, float] = {}

    for summary_path in sorted(reports_root.glob("*/closed_loop_summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        model_path_raw = summary.get("model_path")
        if not model_path_raw:
            continue
        model_path = Path(str(model_path_raw)).resolve()
        if not model_path.exists():
            continue

        tracks = [
            row
            for row in summary.get("tracks", [])
            if bool(row.get("map_available")) and str(row.get("split_name", "")) in {"train", "val"}
        ]
        if not tracks:
            continue

        completion = _mean_metric(tracks, "completion_percent_mean")
        time_to_failure = _mean_metric(tracks, "time_to_failure_sec_mean")
        recovery = _mean_metric(tracks, "recovery_success_rate")
        score = (completion, time_to_failure, recovery)
        if best_score is None or score > best_score:
            best_score = score
            best_model_path = model_path
            best_report_path = summary_path
            best_summary = {
                "seen_val_completion_mean": completion,
                "seen_val_time_to_failure_mean": time_to_failure,
                "seen_val_recovery_success_mean": recovery,
            }

    if best_model_path is None or best_report_path is None:
        raise FileNotFoundError(f"No usable closed-loop report with a valid model_path was found under {reports_root}")
    return best_model_path, best_report_path, best_summary


def discover_track_maps(maps_root: str | Path) -> Dict[str, Path]:
    maps_root = Path(maps_root).resolve()
    candidates: Dict[str, List[Path]] = {}
    for metadata_path in maps_root.glob("*/metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        track_name = str(metadata["track_name"])
        candidates.setdefault(track_name, []).append(metadata_path.parent)

    resolved: Dict[str, Path] = {}
    for track_name, paths in candidates.items():
        resolved[track_name] = sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)[0]
    return resolved


def parse_track_list(values: Sequence[str] | None) -> List[str]:
    if not values:
        return []
    tracks: List[str] = []
    for value in values:
        for chunk in str(value).split(","):
            track = chunk.strip()
            if track:
                tracks.append(track)
    return tracks


def track_split_plan(cfg, train_tracks: Sequence[str] | None = None, val_tracks: Sequence[str] | None = None) -> Dict[str, List[str]]:
    return {
        "train": list(train_tracks) if train_tracks else list(getattr(cfg, "TARGET_POINT_TRAIN_TRACKS", ())),
        "val": list(val_tracks) if val_tracks else list(getattr(cfg, "TARGET_POINT_VAL_TRACKS", ())),
    }


def rollout_thresholds_from_cfg(cfg) -> RolloutCollectionThresholds:
    return RolloutCollectionThresholds(
        first_deviation_cte_m=float(getattr(cfg, "TARGET_POINT_BOOTSTRAP_FIRST_DEVIATION_CTE_M", 0.25)),
        first_deviation_centerline_m=float(getattr(cfg, "TARGET_POINT_BOOTSTRAP_FIRST_DEVIATION_CENTERLINE_M", 0.25)),
        recenter_cte_m=float(getattr(cfg, "TARGET_POINT_BOOTSTRAP_RECENTER_CTE_M", 0.25)),
        recenter_centerline_m=float(getattr(cfg, "TARGET_POINT_BOOTSTRAP_RECENTER_CENTERLINE_M", 0.25)),
        recenter_hold_steps=max(1, int(getattr(cfg, "TARGET_POINT_BOOTSTRAP_RECENTER_HOLD_STEPS", 3))),
        failure_margin_steps=max(1, int(getattr(cfg, "TARGET_POINT_BOOTSTRAP_FAILURE_MARGIN_STEPS", 20))),
    )


def _episode_seed(base_seed: int, split: str, track_index: int, episode_index: int) -> int:
    split_offset = 0 if split == "train" else 10_000
    return int(base_seed) + split_offset + int(track_index) * 100 + int(episode_index)


def _episode_id(split: str, track_name: str, episode_seed: int, episode_index: int) -> str:
    return f"{split}_{_slugify(track_name)}_ep{episode_index:03d}_seed{episode_seed}"


def _load_track_map_by_track(maps_root: str | Path, track_name: str) -> tuple[Path, TrackMapArtifact]:
    map_dirs = discover_track_maps(maps_root)
    if track_name not in map_dirs:
        raise FileNotFoundError(f"No Phase 1 map artifact found for track {track_name!r} under {Path(maps_root).resolve()}")
    map_dir = map_dirs[track_name]
    return map_dir, load_track_map(map_dir)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _save_image(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path, format="JPEG", quality=int(jpeg_quality), optimize=False)


def _is_valid_prediction(cfg, target_x: float, target_y: float) -> bool:
    try:
        target_x = float(target_x)
        target_y = float(target_y)
    except (TypeError, ValueError):
        return False
    return math.isfinite(target_x) and math.isfinite(target_y) and target_y >= float(getattr(cfg, "TARGET_POINT_MIN_FORWARD", 0.0))


def _mark_failure_margin(rows: List[Dict[str, object]], failure_margin_steps: int) -> int:
    if not rows:
        return 0
    start_index = max(0, len(rows) - int(failure_margin_steps))
    marked = 0
    for index in range(start_index, len(rows)):
        if not rows[index].get("failure_margin", False):
            rows[index]["failure_margin"] = True
            rows[index]["scenario"] = "recovery"
            if str(rows[index].get("rollout_event", "none")) == "none":
                rows[index]["rollout_event"] = "failure_margin"
            rows[index]["usable_recovery_sample"] = True
            marked += 1
    return marked


def _sample_row(
    dataset_root: Path,
    split: str,
    track_name: str,
    episode_id: str,
    frame_index: int,
    observation,
    teacher_output: LowNoiseTeacherOutput,
    image_rel_path: str,
    domain_profile: DomainProfile,
    is_closed_map: bool,
    control_hz: float,
) -> Dict[str, object]:
    del dataset_root
    projected = teacher_output.projected_state
    clean_label = teacher_output.clean_label
    action = teacher_output.action
    sample = RawEpisodeSample(
        sample_id=f"{episode_id}_{frame_index:06d}",
        episode_id=episode_id,
        frame_index=int(frame_index),
        elapsed_sec=float(frame_index / max(float(control_hz), 1.0)),
        split=str(split),
        track_name=str(track_name),
        map_id=str(projected.map_id),
        is_closed_map=bool(is_closed_map),
        image_rel_path=str(image_rel_path),
        scenario=str(teacher_output.scenario),
        disturbance_type=str(teacher_output.disturbance_type),
        collection_profile=str(teacher_output.collection_profile),
        domain_profile_id=str(domain_profile.domain_profile_id),
        domain_road_appearance=str(domain_profile.road_appearance),
        domain_edge_appearance=str(domain_profile.edge_appearance),
        domain_environment_appearance=str(domain_profile.environment_appearance),
        domain_brightness=float(domain_profile.brightness),
        domain_contrast=float(domain_profile.contrast),
        domain_saturation=float(domain_profile.saturation),
        domain_rgb_shift_r=int(domain_profile.rgb_shift_r),
        domain_rgb_shift_g=int(domain_profile.rgb_shift_g),
        domain_rgb_shift_b=int(domain_profile.rgb_shift_b),
        domain_blur_radius=float(domain_profile.blur_radius),
        domain_jpeg_quality=int(domain_profile.jpeg_quality),
        pos_x=float(observation.pos[0]),
        pos_y=float(observation.pos[1]),
        pos_z=float(observation.pos[2]),
        yaw_deg=float(observation.yaw_deg),
        pitch_deg=float(observation.pitch_deg),
        roll_deg=float(observation.roll_deg),
        cte=float(observation.cte),
        speed=float(observation.speed),
        forward_vel=float(observation.forward_vel),
        lap_count=int(observation.lap_count),
        last_lap_time=float(observation.last_lap_time),
        hit=str(observation.hit),
        done=bool(observation.done),
        map_distance_m=float(projected.map_distance_m),
        waypoint_index=int(projected.waypoint_index),
        distance_to_centerline_m=float(projected.distance_to_centerline_m),
        centerline_pos_x=float(projected.pos_x),
        centerline_pos_y=float(projected.pos_y),
        centerline_pos_z=float(projected.pos_z),
        centerline_yaw_deg=float(projected.yaw_deg),
        curvature_radpm=float(projected.curvature_radpm),
        delta_heading_2m_deg=float(projected.delta_heading_2m_deg),
        curvature_score=float(projected.curvature_score),
        reference_speed_mps=float(projected.reference_speed_mps),
        teacher_label_mode=str(clean_label.label_mode),
        clean_lookahead_m=float(clean_label.lookahead_m),
        clean_target_x=float(clean_label.target_x),
        clean_target_y=float(clean_label.target_y),
        noise_target_x=float(teacher_output.noise_target_x),
        noise_target_y=float(teacher_output.noise_target_y),
        lane_bias_offset_x=float(teacher_output.lane_bias_offset_x),
        steering_noise=float(teacher_output.steering_noise),
        throttle_noise=float(teacher_output.throttle_noise),
        teacher_commanded_steering=float(teacher_output.teacher_commanded_steering),
        teacher_commanded_throttle=float(teacher_output.teacher_commanded_throttle),
        steering_lag_frames=int(teacher_output.steering_lag_frames),
        safety_override_active=bool(teacher_output.safety_override_active),
        recovery_steering_bias=float(teacher_output.recovery_steering_bias),
        recovery_target_offset_x=float(teacher_output.recovery_target_offset_x),
        recovery_throttle_scale=float(teacher_output.recovery_throttle_scale),
        applied_lookahead_m=float(teacher_output.applied_lookahead_m),
        applied_target_x=float(teacher_output.applied_target_x),
        applied_target_y=float(teacher_output.applied_target_y),
        teacher_steering=float(action.steering),
        teacher_throttle=float(action.throttle),
        teacher_brake=float(action.brake),
        tail_clamped=bool(clean_label.tail_clamped),
    )
    row = asdict(sample)
    row["is_closed_map"] = bool(is_closed_map)
    return row


def _mapping_trace_point_from_observation(observation, action, step_index: int, elapsed_sec: float) -> MapTracePoint:
    return MapTracePoint(
        step_index=int(step_index),
        elapsed_sec=float(elapsed_sec),
        loop_index=0,
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


def collect_generated_roads_episode_from_mapping(
    cfg,
    dataset_root: str | Path,
    split: str,
    track_name: str,
    track_index: int,
    episode_index: int,
    max_steps: int,
    seed: int,
    episode_seed_override: int | None = None,
    label_mode: str = ADAPTIVE_V1,
    collection_profile: str = "phase2_low_noise",
    fixed_throttle: float | None = None,
    spacing_m: float = MAP_SPACING_METERS,
    open_segment_target_distance_m: float = 80.0,
) -> Dict[str, object]:
    dataset_root = Path(dataset_root).resolve()
    episode_seed = int(episode_seed_override) if episode_seed_override is not None else _episode_seed(seed, split, track_index, episode_index)
    teacher_seed = episode_seed + 1_000
    domain_profile = sample_domain_profile(episode_seed + 2_000, track_name, split, episode_index)
    episode_id = f"{split}_{_slugify(track_name)}_ep{episode_index:03d}_seed{episode_seed}"
    episode_dir = dataset_root / "raw" / split / track_name / episode_id
    images_dir = episode_dir / "images"
    map_output_root = dataset_root / "maps" / split / track_name / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    map_output_root.mkdir(parents=True, exist_ok=True)

    resolved_throttle = float(fixed_throttle) if fixed_throttle is not None else 0.10
    mapping_driver = CleanMappingTeacher(
        control_hz=float(getattr(cfg, "DRIVE_LOOP_HZ", 20.0)),
        base_throttle=resolved_throttle,
        min_throttle=resolved_throttle,
        throttle_cte_gain=0.0,
        throttle_steer_gain=0.0,
        max_safe_cte=8.0,
    )
    mapping_driver.reset()

    trace: List[MapTracePoint] = []
    frames: List[tuple[object, object]] = []
    dt = 1.0 / max(float(getattr(cfg, "DRIVE_LOOP_HZ", 20.0)), 1.0)
    travel_distance_m = 0.0
    end_reason = "max_steps"

    with DonkeySimSession(cfg, env_name=track_name, seed=episode_seed) as session:
        correction_sign = mapping_driver.infer_cte_correction_sign(session)
        print(f"[collect] track={track_name} episode={episode_id} mapping_driver_cte_sign={correction_sign}")

        observation = session.prime()
        previous_x, previous_z = float(observation.pos[0]), float(observation.pos[2])
        for frame_index in range(int(max_steps)):
            action = mapping_driver.compute_action(observation)
            trace.append(
                _mapping_trace_point_from_observation(
                    observation=observation,
                    action=action,
                    step_index=frame_index,
                    elapsed_sec=frame_index * dt,
                )
            )
            frames.append((observation, action))

            current_x, _, current_z = observation.pos
            travel_distance_m += math.hypot(float(current_x) - previous_x, float(current_z) - previous_z)
            previous_x, previous_z = float(current_x), float(current_z)

            if observation.hit != "none":
                end_reason = "hit"
                break
            if observation.done:
                raise RuntimeError("Generated-roads mapping collection aborted because the simulator reported done before segment completion.")
            if frame_index >= 200 and travel_distance_m >= float(open_segment_target_distance_m):
                end_reason = "segment_complete"
                break

            observation = session.step(action.steering, action.throttle, action.brake)
        else:
            raise RuntimeError(
                f"Generated-roads mapping collection hit max_steps={max_steps} before covering {open_segment_target_distance_m:.1f}m."
            )

    if len(trace) < 20:
        raise RuntimeError("Generated-roads mapping collection did not capture enough frames to build a usable map.")

    track_map = build_track_map(
        trace,
        track_name=track_name,
        spacing_m=float(spacing_m),
        closed=False,
    )
    map_dir = save_track_map(
        track_map,
        output_root=map_output_root,
        extra_metadata={
            "seed": int(episode_seed),
            "laps_requested": 1,
            "detected_loops": 1,
            "used_open_fallback": True,
            "controller": mapping_driver.summary(),
            "collection_source": "generated_roads_mapping_pass",
        },
    )

    rows: List[Dict[str, object]] = []
    for frame_index, (observation, action) in enumerate(frames):
        projected_state = project_pose_to_centerline(
            track_map,
            pos_x=float(observation.pos[0]),
            pos_y=float(observation.pos[1]),
            pos_z=float(observation.pos[2]),
        )
        clean_label = materialize_label_from_state(
            track_map=track_map,
            label_mode=label_mode,
            map_distance_m=float(projected_state.map_distance_m),
            current_position_xyz=observation.pos,
            yaw_deg=float(observation.yaw_deg),
            speed_mps=max(float(observation.forward_vel), float(observation.speed), 0.0),
            curvature_score=float(projected_state.curvature_score),
            allow_tail_clamp=True,
        )
        if clean_label is None:
            continue

        teacher_output = LowNoiseTeacherOutput(
            action=action,
            projected_state=projected_state,
            clean_label=clean_label,
            collection_profile=str(collection_profile),
            noise_target_x=0.0,
            noise_target_y=0.0,
            lane_bias_offset_x=0.0,
            steering_noise=0.0,
            throttle_noise=0.0,
            teacher_commanded_steering=float(action.steering),
            teacher_commanded_throttle=float(action.throttle),
            steering_lag_frames=0,
            safety_override_active=False,
            recovery_steering_bias=0.0,
            recovery_target_offset_x=0.0,
            recovery_throttle_scale=1.0,
            applied_target_x=float(clean_label.target_x),
            applied_target_y=float(clean_label.target_y),
            applied_lookahead_m=float(clean_label.lookahead_m),
            scenario="nominal",
            disturbance_type="none",
        )

        image_rel_path = str((episode_dir / "images" / f"frame_{frame_index:06d}.jpg").relative_to(dataset_root))
        randomized_image = apply_domain_profile(observation.image, domain_profile)
        _save_image(dataset_root / image_rel_path, randomized_image, domain_profile.jpeg_quality)
        rows.append(
            _sample_row(
                dataset_root=dataset_root,
                split=split,
                track_name=track_name,
                episode_id=episode_id,
                frame_index=frame_index,
                observation=observation,
                teacher_output=teacher_output,
                image_rel_path=image_rel_path,
                domain_profile=domain_profile,
                is_closed_map=False,
                control_hz=float(getattr(cfg, "DRIVE_LOOP_HZ", 20.0)),
            )
        )

    if not rows:
        raise RuntimeError("Generated-roads mapping collection produced zero labeled rows.")

    raw_path = episode_dir / "samples_raw.jsonl"
    _write_jsonl(raw_path, rows)
    metadata = {
        "episode_id": episode_id,
        "split": split,
        "track_name": track_name,
        "map_id": track_map.map_id,
        "map_dir": str(map_dir),
        "is_closed_map": False,
        "seed": int(episode_seed),
        "teacher_seed": int(teacher_seed),
        "domain_profile": domain_profile.to_dict(),
        "teacher_profile": {
            "label_mode": str(label_mode),
            "collection_profile": str(collection_profile),
            "nominal_only": True,
            "source": "generated_roads_mapping_pass",
        },
        "control_source": "clean_mapping_teacher",
        "driver_profile": mapping_driver.summary(),
        "teacher_label_mode": label_mode,
        "collection_profile": str(collection_profile),
        "nominal_only_track": True,
        "sample_count": len(rows),
        "nominal_count": len(rows),
        "recovery_count": 0,
        "progress_m": float(max(float(row["map_distance_m"]) for row in rows)),
        "total_length_m": float(track_map.total_length_m),
        "end_reason": end_reason,
        "raw_samples_jsonl": str(raw_path),
    }
    metadata_path = episode_dir / "episode_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def _rollout_sample_row(
    cfg,
    split: str,
    track_name: str,
    episode_id: str,
    frame_index: int,
    observation,
    track_map: TrackMapArtifact,
    image_rel_path: str,
    domain_profile: DomainProfile,
    label_mode: str,
    driver_model_path: Path,
    reference_controller: TargetPointController,
    model_target_x: float,
    model_target_y: float,
    model_steering: float,
    model_throttle: float,
    invalid_prediction: bool,
    deviation_active: bool,
    rollout_event_id: int,
) -> Dict[str, object]:
    projected = project_pose_to_centerline(track_map, *observation.pos)
    label = materialize_label_from_state(
        track_map=track_map,
        label_mode=label_mode,
        map_distance_m=float(projected.map_distance_m),
        current_position_xyz=(float(observation.pos[0]), float(observation.pos[1]), float(observation.pos[2])),
        yaw_deg=float(observation.yaw_deg),
        speed_mps=max(float(observation.forward_vel), float(observation.speed), 0.0),
        curvature_score=float(projected.curvature_score),
        allow_tail_clamp=False,
    )
    if label is None:
        clean_lookahead_m = float("nan")
        clean_target_x = float("nan")
        clean_target_y = float("nan")
        teacher_steering = 0.0
        teacher_throttle = 0.0
        tail_clamped = False
    else:
        clean_lookahead_m = float(label.lookahead_m)
        clean_target_x = float(label.target_x)
        clean_target_y = float(label.target_y)
        teacher_steering, teacher_throttle = reference_controller.run(clean_target_x, clean_target_y)
        tail_clamped = bool(label.tail_clamped)

    sample = RawEpisodeSample(
        sample_id=f"{episode_id}_{frame_index:06d}",
        episode_id=episode_id,
        frame_index=int(frame_index),
        elapsed_sec=float(frame_index / max(float(getattr(cfg, 'DRIVE_LOOP_HZ', 20.0)), 1.0)),
        split=str(split),
        track_name=str(track_name),
        map_id=str(track_map.map_id),
        is_closed_map=bool(track_map.is_closed),
        image_rel_path=str(image_rel_path),
        scenario="recovery" if deviation_active else "nominal",
        disturbance_type="model_rollout",
        collection_profile=COLLECTION_PROFILE_PHASE55,
        domain_profile_id=str(domain_profile.domain_profile_id),
        domain_road_appearance=str(domain_profile.road_appearance),
        domain_edge_appearance=str(domain_profile.edge_appearance),
        domain_environment_appearance=str(domain_profile.environment_appearance),
        domain_brightness=float(domain_profile.brightness),
        domain_contrast=float(domain_profile.contrast),
        domain_saturation=float(domain_profile.saturation),
        domain_rgb_shift_r=int(domain_profile.rgb_shift_r),
        domain_rgb_shift_g=int(domain_profile.rgb_shift_g),
        domain_rgb_shift_b=int(domain_profile.rgb_shift_b),
        domain_blur_radius=float(domain_profile.blur_radius),
        domain_jpeg_quality=int(domain_profile.jpeg_quality),
        pos_x=float(observation.pos[0]),
        pos_y=float(observation.pos[1]),
        pos_z=float(observation.pos[2]),
        yaw_deg=float(observation.yaw_deg),
        pitch_deg=float(observation.pitch_deg),
        roll_deg=float(observation.roll_deg),
        cte=float(observation.cte),
        speed=float(observation.speed),
        forward_vel=float(observation.forward_vel),
        lap_count=int(observation.lap_count),
        last_lap_time=float(observation.last_lap_time),
        hit=str(observation.hit),
        done=bool(observation.done),
        map_distance_m=float(projected.map_distance_m),
        waypoint_index=int(projected.waypoint_index),
        distance_to_centerline_m=float(projected.distance_to_centerline_m),
        centerline_pos_x=float(projected.pos_x),
        centerline_pos_y=float(projected.pos_y),
        centerline_pos_z=float(projected.pos_z),
        centerline_yaw_deg=float(projected.yaw_deg),
        curvature_radpm=float(projected.curvature_radpm),
        delta_heading_2m_deg=float(projected.delta_heading_2m_deg),
        curvature_score=float(projected.curvature_score),
        reference_speed_mps=float(projected.reference_speed_mps),
        teacher_label_mode=str(label_mode),
        clean_lookahead_m=clean_lookahead_m,
        clean_target_x=clean_target_x,
        clean_target_y=clean_target_y,
        noise_target_x=0.0,
        noise_target_y=0.0,
        lane_bias_offset_x=0.0,
        steering_noise=0.0,
        throttle_noise=0.0,
        teacher_commanded_steering=float(teacher_steering),
        teacher_commanded_throttle=float(teacher_throttle),
        steering_lag_frames=0,
        safety_override_active=False,
        recovery_steering_bias=0.0,
        recovery_target_offset_x=0.0,
        recovery_throttle_scale=1.0,
        applied_lookahead_m=clean_lookahead_m,
        applied_target_x=clean_target_x,
        applied_target_y=clean_target_y,
        teacher_steering=float(teacher_steering),
        teacher_throttle=float(teacher_throttle),
        teacher_brake=0.0,
        tail_clamped=tail_clamped,
        driver_source="rollout",
        driver_model_path=driver_model_path.as_posix(),
        model_target_x=float(model_target_x),
        model_target_y=float(model_target_y),
        model_steering=float(model_steering),
        model_throttle=float(model_throttle),
        invalid_prediction=bool(invalid_prediction),
        rollout_event="recovery" if deviation_active else "none",
        rollout_event_id=int(rollout_event_id if deviation_active else -1),
        deviation_active=bool(deviation_active),
        failure_margin=False,
        usable_recovery_sample=bool(deviation_active),
    )
    return asdict(sample)


def collect_episode(
    cfg,
    dataset_root: str | Path,
    maps_root: str | Path,
    split: str,
    track_name: str,
    track_index: int,
    episode_index: int,
    max_steps: int,
    seed: int,
    episode_seed_override: int | None = None,
    label_mode: str = ADAPTIVE_V1,
    collection_profile: str = "phase2_low_noise",
    nominal_only_tracks: Sequence[str] | None = None,
    fixed_throttle: float | None = None,
) -> Dict[str, object]:
    dataset_root = Path(dataset_root).resolve()
    map_dir, track_map = _load_track_map_by_track(maps_root, track_name)
    nominal_only_tracks = tuple(str(track) for track in (nominal_only_tracks or ()))
    nominal_only = str(track_name) in nominal_only_tracks
    episode_seed = int(episode_seed_override) if episode_seed_override is not None else _episode_seed(seed, split, track_index, episode_index)
    teacher_seed = episode_seed + 1_000
    domain_profile = sample_domain_profile(episode_seed + 2_000, track_name, split, episode_index)
    episode_id = f"{split}_{_slugify(track_name)}_ep{episode_index:03d}_seed{episode_seed}"
    episode_dir = dataset_root / "raw" / split / track_name / episode_id
    images_dir = episode_dir / "images"
    episode_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    teacher = LowNoiseTargetPointTeacher.for_profile(
        track_map=track_map,
        seed=teacher_seed,
        control_hz=float(getattr(cfg, "DRIVE_LOOP_HZ", 20.0)),
        label_mode=label_mode,
        collection_profile=str(collection_profile),
        nominal_only=nominal_only,
        fixed_throttle=fixed_throttle,
    )
    teacher.reset()
    mapping_driver = None
    use_mapping_driver = nominal_only and str(track_name) == "donkey-generated-roads-v0"
    if use_mapping_driver:
        resolved_throttle = float(fixed_throttle) if fixed_throttle is not None else 0.10
        mapping_driver = CleanMappingTeacher(
            control_hz=float(getattr(cfg, "DRIVE_LOOP_HZ", 20.0)),
            base_throttle=resolved_throttle,
            min_throttle=resolved_throttle,
            throttle_cte_gain=0.0,
            throttle_steer_gain=0.0,
            max_safe_cte=8.0,
        )
        mapping_driver.reset()

    rows: List[Dict[str, object]] = []
    nominal_count = 0
    recovery_count = 0
    progress_m = 0.0
    previous_distance = None
    end_reason = "max_steps"

    with DonkeySimSession(cfg, env_name=track_name, seed=episode_seed) as session:
        if mapping_driver is not None:
            correction_sign = mapping_driver.infer_cte_correction_sign(session)
            print(f"[collect] track={track_name} episode={episode_id} mapping_driver_cte_sign={correction_sign}")
        observation = session.prime()
        for frame_index in range(int(max_steps)):
            teacher_output = teacher.compute_action(observation, frame_index)
            action = mapping_driver.compute_action(observation) if mapping_driver is not None else teacher_output.action
            image_rel_path = str((episode_dir / "images" / f"frame_{frame_index:06d}.jpg").relative_to(dataset_root))
            randomized_image = apply_domain_profile(observation.image, domain_profile)
            _save_image(dataset_root / image_rel_path, randomized_image, domain_profile.jpeg_quality)
            row = _sample_row(
                dataset_root=dataset_root,
                split=split,
                track_name=track_name,
                episode_id=episode_id,
                frame_index=frame_index,
                observation=observation,
                teacher_output=teacher_output,
                image_rel_path=image_rel_path,
                domain_profile=domain_profile,
                is_closed_map=track_map.is_closed,
                control_hz=float(getattr(cfg, "DRIVE_LOOP_HZ", 20.0)),
            )
            rows.append(row)
            if row["scenario"] == "recovery":
                recovery_count += 1
            else:
                nominal_count += 1

            current_distance = float(teacher_output.projected_state.map_distance_m)
            if previous_distance is not None:
                delta = current_distance - previous_distance
                if track_map.is_closed:
                    if delta < -0.5 * track_map.total_length_m:
                        delta += track_map.total_length_m
                    elif delta > 0.5 * track_map.total_length_m:
                        delta -= track_map.total_length_m
                progress_m += max(delta, 0.0)
            previous_distance = current_distance

            if observation.hit != "none":
                end_reason = "hit"
                break
            if observation.done and frame_index >= 80:
                end_reason = "sim_done"
                break
            if track_map.is_closed and frame_index >= 80 and progress_m >= 0.95 * float(track_map.total_length_m):
                end_reason = "lap_complete"
                break
            if (not track_map.is_closed) and frame_index >= 80 and current_distance >= max(0.0, track_map.total_length_m - 0.25):
                end_reason = "segment_complete"
                break

            observation = session.step(action.steering, action.throttle, action.brake)

    raw_path = episode_dir / "samples_raw.jsonl"
    _write_jsonl(raw_path, rows)

    metadata = {
        "episode_id": episode_id,
        "split": split,
        "track_name": track_name,
        "map_id": track_map.map_id,
        "map_dir": str(map_dir),
        "is_closed_map": bool(track_map.is_closed),
        "seed": int(episode_seed),
        "teacher_seed": int(teacher_seed),
        "domain_profile": domain_profile.to_dict(),
        "teacher_profile": teacher.summary(),
        "control_source": "clean_mapping_teacher" if mapping_driver is not None else "target_point_teacher",
        "driver_profile": mapping_driver.summary() if mapping_driver is not None else teacher.summary(),
        "teacher_label_mode": label_mode,
        "collection_profile": str(collection_profile),
        "nominal_only_track": bool(nominal_only),
        "sample_count": len(rows),
        "nominal_count": int(nominal_count),
        "recovery_count": int(recovery_count),
        "progress_m": float(progress_m),
        "total_length_m": float(track_map.total_length_m),
        "end_reason": end_reason,
        "raw_samples_jsonl": str(raw_path),
    }
    metadata_path = episode_dir / "episode_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def collect_dataset(
    cfg,
    maps_root: str | Path,
    dataset_root: str | Path,
    split_tracks: Dict[str, Sequence[str]],
    episodes_per_track: int,
    max_steps: int,
    seed: int,
    label_mode: str = ADAPTIVE_V1,
    collection_profile: str = "phase2_low_noise",
    nominal_only_tracks: Sequence[str] | None = None,
    fixed_throttle: float | None = None,
    min_samples_per_track: int | None = None,
    remap_generated_roads: bool = False,
) -> Dict[str, object]:
    dataset_root = Path(dataset_root).resolve()
    index_root = dataset_root / "index"
    index_root.mkdir(parents=True, exist_ok=True)

    nominal_only_tracks = tuple(str(track) for track in (nominal_only_tracks or ()))
    min_samples_per_track = None if min_samples_per_track is None else max(1, int(min_samples_per_track))
    max_generated_road_map_attempts = 8
    episode_rows: List[Dict[str, object]] = []
    sample_count_by_split_track: Dict[str, Dict[str, int]] = {}
    episode_count_by_split_track: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val"):
        tracks = list(split_tracks.get(split, ()))
        for track_index, track_name in enumerate(tracks):
            episode_index = 0
            collected_samples = 0
            collected_episodes = 0
            while True:
                enough_episodes = episode_index >= int(episodes_per_track)
                enough_samples = min_samples_per_track is None or collected_samples >= min_samples_per_track
                if enough_episodes and enough_samples:
                    break

                base_episode_seed = _episode_seed(seed, split, track_index, episode_index)
                episode_seed = int(base_episode_seed)
                episode_id = _episode_id(split, track_name, episode_seed, episode_index)
                episode_maps_root = maps_root
                if bool(remap_generated_roads) and str(track_name) == "donkey-generated-roads-v0" and str(track_name) in nominal_only_tracks:
                    map_attempt = 0
                    while True:
                        episode_seed = int(base_episode_seed + map_attempt)
                        episode_id = _episode_id(split, track_name, episode_seed, episode_index)
                        try:
                            metadata = collect_generated_roads_episode_from_mapping(
                                cfg=cfg,
                                dataset_root=dataset_root,
                                split=split,
                                track_name=track_name,
                                track_index=track_index,
                                episode_index=episode_index,
                                max_steps=max_steps,
                                seed=seed,
                                episode_seed_override=episode_seed,
                                label_mode=label_mode,
                                collection_profile=collection_profile,
                                fixed_throttle=fixed_throttle,
                            )
                            break
                        except RuntimeError as exc:
                            map_attempt += 1
                            if map_attempt >= max_generated_road_map_attempts:
                                raise RuntimeError(
                                    f"Failed to build a generated-roads map for split={split} track={track_name} "
                                    f"episode_index={episode_index} after {max_generated_road_map_attempts} attempts."
                                ) from exc
                            print(
                                f"[collect] remap_retry split={split} track={track_name} episode_index={episode_index} "
                                f"seed={episode_seed} reason={exc}"
                            )
                else:
                    if bool(remap_generated_roads) and str(track_name) == "donkey-generated-roads-v0":
                        map_attempt = 0
                        while True:
                            episode_seed = int(base_episode_seed + map_attempt)
                            episode_id = _episode_id(split, track_name, episode_seed, episode_index)
                            episode_maps_root = dataset_root / "maps" / split / track_name / episode_id
                            try:
                                build_phase1_map(
                                    cfg=cfg,
                                    track_name=str(track_name),
                                    output_root=episode_maps_root,
                                    seed=int(episode_seed),
                                    laps=1,
                                    max_steps=max_steps,
                                )
                                break
                            except RuntimeError as exc:
                                map_attempt += 1
                                if map_attempt >= max_generated_road_map_attempts:
                                    raise RuntimeError(
                                        f"Failed to build a generated-roads map for split={split} track={track_name} "
                                        f"episode_index={episode_index} after {max_generated_road_map_attempts} attempts."
                                    ) from exc
                                print(
                                    f"[collect] remap_retry split={split} track={track_name} episode_index={episode_index} "
                                    f"seed={episode_seed} reason={exc}"
                                )

                    metadata = collect_episode(
                        cfg=cfg,
                        dataset_root=dataset_root,
                        maps_root=episode_maps_root,
                        split=split,
                        track_name=track_name,
                        track_index=track_index,
                        episode_index=episode_index,
                        max_steps=max_steps,
                        seed=seed,
                        episode_seed_override=episode_seed,
                        label_mode=label_mode,
                        collection_profile=collection_profile,
                        nominal_only_tracks=nominal_only_tracks,
                        fixed_throttle=fixed_throttle,
                    )
                sample_count = int(metadata["sample_count"])
                if sample_count <= 0:
                    raise RuntimeError(
                        f"Collection produced zero samples for split={split} track={track_name} episode={metadata['episode_id']}"
                    )
                episode_rows.append(metadata)
                collected_samples += sample_count
                collected_episodes += 1
                episode_index += 1
                sample_count_by_split_track.setdefault(split, {})[str(track_name)] = int(collected_samples)
                episode_count_by_split_track.setdefault(split, {})[str(track_name)] = int(collected_episodes)
                print(
                    f"[collect] split={split} track={track_name} episode={metadata['episode_id']} "
                    f"samples={metadata['sample_count']} recovery={metadata['recovery_count']} "
                    f"profile={metadata['collection_profile']} nominal_only={metadata['nominal_only_track']} "
                    f"end_reason={metadata['end_reason']} "
                    f"track_samples={collected_samples}"
                )

    raw_index_path = index_root / "raw_episodes.jsonl"
    _write_jsonl(raw_index_path, episode_rows)
    summary = {
        "dataset_root": str(dataset_root),
        "raw_index_path": str(raw_index_path),
        "episode_count": len(episode_rows),
        "episodes_per_track_minimum": int(episodes_per_track),
        "min_samples_per_track": min_samples_per_track,
        "remap_generated_roads": bool(remap_generated_roads),
        "sample_count_by_split_track": sample_count_by_split_track,
        "episode_count_by_split_track": episode_count_by_split_track,
    }
    summary_path = index_root / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        **summary,
        "summary_path": str(summary_path),
    }


def collect_model_rollout_episode(
    cfg,
    dataset_root: str | Path,
    maps_root: str | Path,
    track_name: str,
    track_index: int,
    episode_index: int,
    max_steps: int,
    seed: int,
    driver_model_path: str | Path,
    label_mode: str = ADAPTIVE_V1,
    nominal_only_tracks: Sequence[str] | None = None,
    rollout_thresholds: Optional[RolloutCollectionThresholds] = None,
) -> Dict[str, object]:
    from target_point.pilot import TargetPointPilot

    dataset_root = Path(dataset_root).resolve()
    driver_model_path = Path(driver_model_path).resolve()
    map_dir, track_map = _load_track_map_by_track(maps_root, track_name)
    nominal_only_tracks = tuple(str(track) for track in (nominal_only_tracks or ()))
    nominal_only = str(track_name) in nominal_only_tracks
    thresholds = rollout_thresholds or rollout_thresholds_from_cfg(cfg)

    episode_seed = _episode_seed(seed, "train", track_index, episode_index)
    domain_profile = sample_domain_profile(episode_seed + 2_000, track_name, "train", episode_index)
    episode_id = f"train_{_slugify(track_name)}_rollout_ep{episode_index:03d}_seed{episode_seed}"
    episode_dir = dataset_root / "raw" / "train" / track_name / episode_id
    images_dir = episode_dir / "images"
    episode_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    pilot = TargetPointPilot(cfg)
    pilot.load(driver_model_path.as_posix())
    controller = TargetPointController(cfg)
    reference_controller = TargetPointController(cfg)

    rows: List[Dict[str, object]] = []
    deviation_active = False
    rollout_event_id = 0
    recenter_hold = 0
    usable_recovery_count = 0
    failure_margin_count = 0
    progress_m = 0.0
    previous_distance = None
    end_reason = "max_steps"
    offtrack_hold = 0
    offtrack_cte_m = float(getattr(cfg, "TARGET_POINT_EVAL_OFFTRACK_CTE_M", 1.0))
    offtrack_centerline_m = float(getattr(cfg, "TARGET_POINT_EVAL_OFFTRACK_CENTERLINE_M", 1.1))
    offtrack_hold_steps = max(1, int(getattr(cfg, "TARGET_POINT_EVAL_OFFTRACK_HOLD_STEPS", 3)))

    try:
        with DonkeySimSession(cfg, env_name=track_name, seed=episode_seed) as session:
            observation = session.prime()
            for frame_index in range(int(max_steps)):
                projected = project_pose_to_centerline(track_map, *observation.pos)
                centerline_error = float(projected.distance_to_centerline_m)
                current_distance = float(projected.map_distance_m)
                center_error = max(abs(float(observation.cte)), centerline_error)
                is_first_deviation = (not nominal_only) and thresholds.is_first_deviation(float(observation.cte), centerline_error)
                is_recentered = thresholds.is_recentered(float(observation.cte), centerline_error)

                if is_first_deviation and not deviation_active:
                    deviation_active = True
                    rollout_event_id += 1
                    recenter_hold = 0
                elif deviation_active:
                    if is_recentered:
                        recenter_hold += 1
                        if recenter_hold >= thresholds.recenter_hold_steps:
                            deviation_active = False
                            recenter_hold = 0
                    else:
                        recenter_hold = 0

                model_target_x, model_target_y = pilot.run(observation.image)
                model_steering, model_throttle = controller.run(model_target_x, model_target_y)
                invalid_prediction = not _is_valid_prediction(cfg, model_target_x, model_target_y)

                image_rel_path = str((episode_dir / "images" / f"frame_{frame_index:06d}.jpg").relative_to(dataset_root))
                randomized_image = apply_domain_profile(observation.image, domain_profile)
                _save_image(dataset_root / image_rel_path, randomized_image, domain_profile.jpeg_quality)
                row = _rollout_sample_row(
                    cfg=cfg,
                    split="train",
                    track_name=track_name,
                    episode_id=episode_id,
                    frame_index=frame_index,
                    observation=observation,
                    track_map=track_map,
                    image_rel_path=image_rel_path,
                    domain_profile=domain_profile,
                    label_mode=label_mode,
                    driver_model_path=driver_model_path,
                    reference_controller=reference_controller,
                    model_target_x=float(model_target_x),
                    model_target_y=float(model_target_y),
                    model_steering=float(model_steering),
                    model_throttle=float(model_throttle),
                    invalid_prediction=bool(invalid_prediction),
                    deviation_active=bool(deviation_active),
                    rollout_event_id=int(rollout_event_id),
                )
                rows.append(row)
                if bool(row["usable_recovery_sample"]):
                    usable_recovery_count += 1

                if previous_distance is not None:
                    delta = current_distance - previous_distance
                    if track_map.is_closed:
                        if delta < -0.5 * track_map.total_length_m:
                            delta += track_map.total_length_m
                        elif delta > 0.5 * track_map.total_length_m:
                            delta -= track_map.total_length_m
                    progress_m += max(delta, 0.0)
                previous_distance = current_distance

                if abs(float(observation.cte)) >= offtrack_cte_m or centerline_error >= offtrack_centerline_m:
                    offtrack_hold += 1
                else:
                    offtrack_hold = 0

                if observation.hit != "none":
                    end_reason = "hit"
                    break
                if observation.done and frame_index >= 80:
                    end_reason = "sim_done"
                    break
                if offtrack_hold >= offtrack_hold_steps:
                    end_reason = "offtrack"
                    break
                if track_map.is_closed and frame_index >= 80 and progress_m >= 0.95 * float(track_map.total_length_m):
                    end_reason = "lap_complete"
                    break
                if (not track_map.is_closed) and frame_index >= 80 and current_distance >= max(0.0, track_map.total_length_m - 0.25):
                    end_reason = "segment_complete"
                    break

                observation = session.step(model_steering, model_throttle, 0.0)
    finally:
        pilot.shutdown()

    if end_reason in {"offtrack", "hit", "sim_done"} and not nominal_only:
        failure_margin_count = _mark_failure_margin(rows, thresholds.failure_margin_steps)
        usable_recovery_count = sum(1 for row in rows if bool(row.get("usable_recovery_sample", False)))

    raw_path = episode_dir / "samples_raw.jsonl"
    _write_jsonl(raw_path, rows)

    metadata = {
        "episode_id": episode_id,
        "split": "train",
        "track_name": track_name,
        "map_id": track_map.map_id,
        "map_dir": str(map_dir),
        "is_closed_map": bool(track_map.is_closed),
        "seed": int(episode_seed),
        "driver_model_path": driver_model_path.as_posix(),
        "collection_profile": COLLECTION_PROFILE_PHASE55,
        "teacher_label_mode": label_mode,
        "nominal_only_track": bool(nominal_only),
        "sample_count": len(rows),
        "nominal_count": int(sum(1 for row in rows if str(row["scenario"]) != "recovery")),
        "recovery_count": int(sum(1 for row in rows if str(row["scenario"]) == "recovery")),
        "usable_recovery_count": int(usable_recovery_count),
        "failure_margin_count": int(failure_margin_count),
        "progress_m": float(progress_m),
        "total_length_m": float(track_map.total_length_m),
        "end_reason": end_reason,
        "raw_samples_jsonl": str(raw_path),
    }
    metadata_path = episode_dir / "episode_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def collect_rollout_dataset(
    cfg,
    maps_root: str | Path,
    dataset_root: str | Path,
    train_tracks: Sequence[str],
    max_episodes_per_track: int,
    max_steps: int,
    seed: int,
    driver_model_path: str | Path | None = None,
    label_mode: str = ADAPTIVE_V1,
    nominal_only_tracks: Sequence[str] | None = None,
    min_usable_recovery_samples: Optional[int] = None,
) -> Dict[str, object]:
    dataset_root = Path(dataset_root).resolve()
    index_root = dataset_root / "index"
    index_root.mkdir(parents=True, exist_ok=True)
    rollout_thresholds = rollout_thresholds_from_cfg(cfg)
    nominal_only_tracks = tuple(str(track) for track in (nominal_only_tracks or ()))
    target_recovery_samples = int(
        min_usable_recovery_samples
        if min_usable_recovery_samples is not None
        else getattr(cfg, "TARGET_POINT_BOOTSTRAP_MIN_USABLE_RECOVERY_SAMPLES", 120)
    )

    selected_model_summary = {}
    selected_report_path = None
    if driver_model_path is None:
        resolved_model_path, selected_report_path, selected_model_summary = select_best_rollout_driver_model(cfg)
        driver_model_path = resolved_model_path
    else:
        driver_model_path = Path(driver_model_path).resolve()

    episode_rows: List[Dict[str, object]] = []
    track_totals: Dict[str, int] = {str(track_name): 0 for track_name in train_tracks}
    for track_index, track_name in enumerate(train_tracks):
        for episode_index in range(int(max_episodes_per_track)):
            metadata = collect_model_rollout_episode(
                cfg=cfg,
                dataset_root=dataset_root,
                maps_root=maps_root,
                track_name=str(track_name),
                track_index=track_index,
                episode_index=episode_index,
                max_steps=max_steps,
                seed=seed,
                driver_model_path=driver_model_path,
                label_mode=label_mode,
                nominal_only_tracks=nominal_only_tracks,
                rollout_thresholds=rollout_thresholds,
            )
            episode_rows.append(metadata)
            track_totals[str(track_name)] += int(metadata["usable_recovery_count"])
            print(
                f"[rollout] track={track_name} episode={metadata['episode_id']} "
                f"samples={metadata['sample_count']} usable_recovery={metadata['usable_recovery_count']} "
                f"failure_margin={metadata['failure_margin_count']} end_reason={metadata['end_reason']}"
            )
            if track_totals[str(track_name)] >= target_recovery_samples:
                break

    raw_index_path = index_root / "raw_episodes.jsonl"
    _write_jsonl(raw_index_path, episode_rows)
    summary = {
        "dataset_root": str(dataset_root),
        "raw_index_path": str(raw_index_path),
        "episode_count": len(episode_rows),
        "driver_model_path": Path(driver_model_path).resolve().as_posix(),
        "driver_report_path": None if selected_report_path is None else selected_report_path.as_posix(),
        "driver_report_summary": selected_model_summary,
        "target_usable_recovery_samples_per_track": int(target_recovery_samples),
        "usable_recovery_samples_by_track": track_totals,
    }
    summary_path = index_root / "rollout_collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
