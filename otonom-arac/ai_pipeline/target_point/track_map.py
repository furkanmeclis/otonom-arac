"""Track mapping artifacts and geometry helpers for target-point labeling."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


MAP_SPACING_METERS = 0.25
LOOKAHEAD_HEADING_METERS = 2.0


@dataclass(frozen=True)
class MapTracePoint:
    step_index: int
    elapsed_sec: float
    loop_index: int
    lap_count: int
    pos_x: float
    pos_y: float
    pos_z: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    cte: float
    speed: float
    forward_vel: float
    steering: float
    throttle: float
    brake: float
    hit: str


@dataclass(frozen=True)
class CenterlineWaypoint:
    waypoint_index: int
    distance_m: float
    pos_x: float
    pos_y: float
    pos_z: float
    heading_x: float
    heading_z: float
    yaw_deg: float
    curvature_radpm: float
    delta_heading_2m_deg: float
    curvature_score: float
    reference_speed_mps: float


@dataclass(frozen=True)
class TrackMapArtifact:
    map_id: str
    track_name: str
    geometry_hash: str
    is_closed: bool
    spacing_m: float
    total_length_m: float
    source_lap_count: int
    raw_trace: List[MapTracePoint]
    centerline: List[CenterlineWaypoint]


TRACE_FIELDNAMES = list(MapTracePoint.__dataclass_fields__.keys())
WAYPOINT_FIELDNAMES = list(CenterlineWaypoint.__dataclass_fields__.keys())


def _wrap_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _loop_lengths(points_xz: np.ndarray) -> Tuple[np.ndarray, float]:
    points_ext = np.vstack([points_xz, points_xz[0]])
    segment_lengths = np.linalg.norm(np.diff(points_ext, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    return segment_lengths, total_length


def _closed_cumulative_distances(points_xz: np.ndarray) -> Tuple[np.ndarray, float]:
    segment_lengths, total_length = _loop_lengths(points_xz)
    cumulative = np.zeros(len(points_xz), dtype=np.float32)
    if len(points_xz) > 1:
        cumulative[1:] = np.cumsum(segment_lengths[:-1], dtype=np.float32)
    return cumulative, total_length


def _open_cumulative_distances(points_xz: np.ndarray) -> Tuple[np.ndarray, float]:
    cumulative = np.zeros(len(points_xz), dtype=np.float32)
    if len(points_xz) > 1:
        cumulative[1:] = np.cumsum(np.linalg.norm(np.diff(points_xz, axis=0), axis=1), dtype=np.float32)
    return cumulative, float(cumulative[-1]) if len(cumulative) else 0.0


def _dedupe_lap_trace(trace: Sequence[MapTracePoint], min_step_distance: float = 0.02) -> List[MapTracePoint]:
    if not trace:
        return []

    deduped = [trace[0]]
    last = trace[0]
    for point in trace[1:]:
        dx = float(point.pos_x) - float(last.pos_x)
        dz = float(point.pos_z) - float(last.pos_z)
        if math.hypot(dx, dz) >= float(min_step_distance):
            deduped.append(point)
            last = point
    return deduped


def split_trace_into_laps(trace: Sequence[MapTracePoint]) -> List[List[MapTracePoint]]:
    lap_groups: Dict[int, List[MapTracePoint]] = {}
    for point in trace:
        lap_groups.setdefault(int(point.loop_index), []).append(point)

    if not lap_groups:
        raise ValueError("Cannot build a track map from an empty trace.")

    full_laps = []
    for lap_idx in sorted(lap_groups.keys()):
        lap_trace = _dedupe_lap_trace(lap_groups[lap_idx])
        if len(lap_trace) >= 20:
            full_laps.append(lap_trace)

    if not full_laps:
        fallback = _dedupe_lap_trace(trace)
        if len(fallback) < 20:
            raise ValueError("Trace did not contain enough usable points to build a track map.")
        full_laps = [fallback]

    return full_laps


def _resample_closed_loop(
    points_xyz: np.ndarray,
    speeds: np.ndarray,
    spacing_m: float,
    target_count: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    points_xz = points_xyz[:, [0, 2]]
    segment_lengths, total_length = _loop_lengths(points_xz)
    if total_length <= 0.0:
        raise ValueError("Cannot resample a degenerate lap with zero length.")

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    xyz_ext = np.vstack([points_xyz, points_xyz[0]])
    speed_ext = np.concatenate([speeds, speeds[:1]])

    if target_count is None:
        target_count = max(4, int(round(total_length / float(spacing_m))))

    sample_distances = np.linspace(0.0, total_length, int(target_count), endpoint=False, dtype=np.float32)
    sampled_xyz = np.stack(
        [np.interp(sample_distances, cumulative, xyz_ext[:, axis]) for axis in range(3)],
        axis=1,
    ).astype(np.float32)
    sampled_speeds = np.interp(sample_distances, cumulative, speed_ext).astype(np.float32)
    return sampled_xyz, sampled_speeds, total_length


def _resample_open_path(
    points_xyz: np.ndarray,
    speeds: np.ndarray,
    spacing_m: float,
    target_count: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    points_xz = points_xyz[:, [0, 2]]
    cumulative, total_length = _open_cumulative_distances(points_xz)
    if total_length <= 0.0:
        raise ValueError("Cannot resample a degenerate open path with zero length.")

    if target_count is None:
        target_count = max(2, int(round(total_length / float(spacing_m))) + 1)

    sample_distances = np.linspace(0.0, total_length, int(target_count), endpoint=True, dtype=np.float32)
    sampled_xyz = np.stack(
        [np.interp(sample_distances, cumulative, points_xyz[:, axis]) for axis in range(3)],
        axis=1,
    ).astype(np.float32)
    sampled_speeds = np.interp(sample_distances, cumulative, speeds).astype(np.float32)
    return sampled_xyz, sampled_speeds, total_length


def _align_to_reference(reference_xyz: np.ndarray, candidate_xyz: np.ndarray, candidate_speeds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    reference_origin = reference_xyz[0, [0, 2]]
    candidate_xz = candidate_xyz[:, [0, 2]]
    shift = int(np.argmin(np.linalg.norm(candidate_xz - reference_origin, axis=1)))
    return np.roll(candidate_xyz, -shift, axis=0), np.roll(candidate_speeds, -shift, axis=0)


def _heading_vectors(points_xyz: np.ndarray) -> np.ndarray:
    points_xz = points_xyz[:, [0, 2]]
    prev_points = np.roll(points_xz, 1, axis=0)
    next_points = np.roll(points_xz, -1, axis=0)
    directions = next_points - prev_points
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    return directions / norms


def _heading_vectors_open(points_xyz: np.ndarray) -> np.ndarray:
    n_points = len(points_xyz)
    points_xz = points_xyz[:, [0, 2]]
    directions = np.zeros((n_points, 2), dtype=np.float32)
    if n_points == 1:
        directions[0] = np.asarray([0.0, 1.0], dtype=np.float32)
        return directions

    directions[0] = points_xz[1] - points_xz[0]
    directions[-1] = points_xz[-1] - points_xz[-2]
    if n_points > 2:
        directions[1:-1] = points_xz[2:] - points_xz[:-2]

    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    return directions / norms


def _heading_angles_deg(headings: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(headings[:, 0], headings[:, 1])).astype(np.float32)


def _advance_index_by_distance(cumulative: np.ndarray, total_length: float, index: int, distance_ahead: float) -> int:
    target_distance = (float(cumulative[index]) + float(distance_ahead)) % float(total_length)
    return int(np.argmin(np.abs(cumulative - target_distance)))


def _curvature_values(points_xyz: np.ndarray, cumulative: np.ndarray, total_length: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    headings = _heading_vectors(points_xyz)
    heading_angles = np.unwrap(np.arctan2(headings[:, 0], headings[:, 1]))
    n_points = len(points_xyz)
    curvature = np.zeros(n_points, dtype=np.float32)
    delta_heading_2m_deg = np.zeros(n_points, dtype=np.float32)

    for index in range(n_points):
        prev_index = (index - 1) % n_points
        next_index = (index + 1) % n_points
        prev_distance = float(np.linalg.norm(points_xyz[index, [0, 2]] - points_xyz[prev_index, [0, 2]]))
        next_distance = float(np.linalg.norm(points_xyz[next_index, [0, 2]] - points_xyz[index, [0, 2]]))
        span = max(prev_distance + next_distance, 1e-6)
        curvature[index] = float((heading_angles[next_index] - heading_angles[prev_index]) / span)

        future_index = _advance_index_by_distance(cumulative, total_length, index, LOOKAHEAD_HEADING_METERS)
        delta_heading_deg = math.degrees(float(heading_angles[future_index] - heading_angles[index]))
        delta_heading_2m_deg[index] = _wrap_angle_deg(delta_heading_deg)

    curvature_score = np.clip(np.abs(delta_heading_2m_deg) / 35.0, 0.0, 1.0).astype(np.float32)
    return curvature, delta_heading_2m_deg, curvature_score


def _curvature_values_open(points_xyz: np.ndarray, cumulative: np.ndarray, total_length: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    headings = _heading_vectors_open(points_xyz)
    heading_angles = np.unwrap(np.arctan2(headings[:, 0], headings[:, 1]))
    n_points = len(points_xyz)
    curvature = np.zeros(n_points, dtype=np.float32)
    delta_heading_2m_deg = np.zeros(n_points, dtype=np.float32)

    for index in range(n_points):
        prev_index = max(index - 1, 0)
        next_index = min(index + 1, n_points - 1)
        prev_distance = float(np.linalg.norm(points_xyz[index, [0, 2]] - points_xyz[prev_index, [0, 2]]))
        next_distance = float(np.linalg.norm(points_xyz[next_index, [0, 2]] - points_xyz[index, [0, 2]]))
        span = max(prev_distance + next_distance, 1e-6)
        curvature[index] = float((heading_angles[next_index] - heading_angles[prev_index]) / span)

        target_distance = min(float(cumulative[index]) + LOOKAHEAD_HEADING_METERS, float(total_length))
        future_index = int(np.searchsorted(cumulative, target_distance, side="left"))
        future_index = min(max(future_index, index), n_points - 1)
        delta_heading_deg = math.degrees(float(heading_angles[future_index] - heading_angles[index]))
        delta_heading_2m_deg[index] = _wrap_angle_deg(delta_heading_deg)

    curvature_score = np.clip(np.abs(delta_heading_2m_deg) / 35.0, 0.0, 1.0).astype(np.float32)
    return curvature, delta_heading_2m_deg, curvature_score


def _geometry_hash(points_xyz: np.ndarray) -> str:
    rounded = np.round(points_xyz[:, [0, 2]], decimals=3)
    digest = hashlib.sha1(rounded.astype(np.float32).tobytes()).hexdigest()
    return digest


def build_track_map(
    trace: Sequence[MapTracePoint],
    track_name: str,
    spacing_m: float = MAP_SPACING_METERS,
    closed: bool = True,
) -> TrackMapArtifact:
    laps = split_trace_into_laps(trace)

    lap_xyz_list = []
    lap_speed_list = []
    lap_lengths = []
    for lap in laps:
        lap_xyz = np.asarray([[point.pos_x, point.pos_y, point.pos_z] for point in lap], dtype=np.float32)
        lap_speeds = np.asarray([point.forward_vel if point.forward_vel > 0.0 else point.speed for point in lap], dtype=np.float32)
        lap_xyz_list.append(lap_xyz)
        lap_speed_list.append(lap_speeds)
        if closed:
            lap_lengths.append(float(np.sum(np.linalg.norm(np.diff(np.vstack([lap_xyz[:, [0, 2]], lap_xyz[0, [0, 2]]]), axis=0), axis=1))))
        else:
            lap_lengths.append(float(np.sum(np.linalg.norm(np.diff(lap_xyz[:, [0, 2]], axis=0), axis=1))))

    target_count = max(2 if not closed else 4, int(round(float(np.mean(lap_lengths)) / float(spacing_m))) + (0 if closed else 1))

    aligned_xyz = []
    aligned_speeds = []
    for index, lap_xyz in enumerate(lap_xyz_list):
        if closed:
            sampled_xyz, sampled_speeds, _ = _resample_closed_loop(
                lap_xyz,
                lap_speed_list[index],
                spacing_m=spacing_m,
                target_count=target_count,
            )
        else:
            sampled_xyz, sampled_speeds, _ = _resample_open_path(
                lap_xyz,
                lap_speed_list[index],
                spacing_m=spacing_m,
                target_count=target_count,
            )

        if index == 0 or not closed:
            aligned_xyz.append(sampled_xyz)
            aligned_speeds.append(sampled_speeds)
        else:
            xyz_aligned, speed_aligned = _align_to_reference(aligned_xyz[0], sampled_xyz, sampled_speeds)
            aligned_xyz.append(xyz_aligned)
            aligned_speeds.append(speed_aligned)

    mean_xyz = np.mean(np.stack(aligned_xyz, axis=0), axis=0).astype(np.float32)
    mean_speeds = np.mean(np.stack(aligned_speeds, axis=0), axis=0).astype(np.float32)
    if closed:
        cumulative, total_length = _closed_cumulative_distances(mean_xyz[:, [0, 2]])
        headings = _heading_vectors(mean_xyz)
        curvature, delta_heading_2m_deg, curvature_score = _curvature_values(mean_xyz, cumulative, total_length)
    else:
        cumulative, total_length = _open_cumulative_distances(mean_xyz[:, [0, 2]])
        headings = _heading_vectors_open(mean_xyz)
        curvature, delta_heading_2m_deg, curvature_score = _curvature_values_open(mean_xyz, cumulative, total_length)
    yaw_deg = _heading_angles_deg(headings)

    geometry_hash = _geometry_hash(mean_xyz)
    map_id = f"{track_name}_{geometry_hash[:10]}"

    centerline = []
    for index in range(len(mean_xyz)):
        centerline.append(
            CenterlineWaypoint(
                waypoint_index=int(index),
                distance_m=float(cumulative[index]),
                pos_x=float(mean_xyz[index, 0]),
                pos_y=float(mean_xyz[index, 1]),
                pos_z=float(mean_xyz[index, 2]),
                heading_x=float(headings[index, 0]),
                heading_z=float(headings[index, 1]),
                yaw_deg=float(yaw_deg[index]),
                curvature_radpm=float(curvature[index]),
                delta_heading_2m_deg=float(delta_heading_2m_deg[index]),
                curvature_score=float(curvature_score[index]),
                reference_speed_mps=float(mean_speeds[index]),
            )
        )

    return TrackMapArtifact(
        map_id=map_id,
        track_name=track_name,
        geometry_hash=geometry_hash,
        is_closed=bool(closed),
        spacing_m=float(spacing_m),
        total_length_m=float(total_length),
        source_lap_count=len(laps),
        raw_trace=list(trace),
        centerline=centerline,
    )


def centerline_arrays(centerline: Sequence[CenterlineWaypoint]) -> Dict[str, np.ndarray]:
    return {
        "distance_m": np.asarray([point.distance_m for point in centerline], dtype=np.float32),
        "pos_x": np.asarray([point.pos_x for point in centerline], dtype=np.float32),
        "pos_y": np.asarray([point.pos_y for point in centerline], dtype=np.float32),
        "pos_z": np.asarray([point.pos_z for point in centerline], dtype=np.float32),
    }


def _clip_distance(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def interpolate_centerline_position(
    centerline: Sequence[CenterlineWaypoint],
    total_length_m: float,
    distance_m: float,
    is_closed: bool = True,
) -> np.ndarray:
    arrays = centerline_arrays(centerline)
    base_distances = arrays["distance_m"]
    if is_closed:
        extended_distances = np.concatenate([base_distances, np.asarray([float(total_length_m)], dtype=np.float32)])
        normalized_distance = float(distance_m) % float(total_length_m)
    else:
        extended_distances = base_distances
        normalized_distance = _clip_distance(float(distance_m), 0.0, float(total_length_m))

    coords = []
    for key in ("pos_x", "pos_y", "pos_z"):
        extended_values = np.concatenate([arrays[key], arrays[key][:1]]) if is_closed else arrays[key]
        coords.append(float(np.interp(normalized_distance, extended_distances, extended_values)))
    return np.asarray(coords, dtype=np.float32)


def save_track_map(artifact: TrackMapArtifact, output_root: str | Path, extra_metadata: Dict[str, object] | None = None) -> Path:
    output_dir = Path(output_root) / artifact.map_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "map_id": artifact.map_id,
        "track_name": artifact.track_name,
        "geometry_hash": artifact.geometry_hash,
        "is_closed": artifact.is_closed,
        "spacing_m": artifact.spacing_m,
        "total_length_m": artifact.total_length_m,
        "source_lap_count": artifact.source_lap_count,
        "raw_trace_count": len(artifact.raw_trace),
        "centerline_count": len(artifact.centerline),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)

    with (output_dir / "raw_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDNAMES)
        writer.writeheader()
        for point in artifact.raw_trace:
            writer.writerow(asdict(point))

    with (output_dir / "centerline.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=WAYPOINT_FIELDNAMES)
        writer.writeheader()
        for point in artifact.centerline:
            writer.writerow(asdict(point))

    return output_dir


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return [dict(row) for row in reader]


def load_track_map(map_dir: str | Path) -> TrackMapArtifact:
    map_dir = Path(map_dir)
    metadata = json.loads((map_dir / "metadata.json").read_text(encoding="utf-8"))
    raw_trace_rows = _load_csv_rows(map_dir / "raw_trace.csv")
    centerline_rows = _load_csv_rows(map_dir / "centerline.csv")

    raw_trace = [
        MapTracePoint(
            step_index=int(row["step_index"]),
            elapsed_sec=float(row["elapsed_sec"]),
            loop_index=int(row["loop_index"]),
            lap_count=int(row["lap_count"]),
            pos_x=float(row["pos_x"]),
            pos_y=float(row["pos_y"]),
            pos_z=float(row["pos_z"]),
            yaw_deg=float(row["yaw_deg"]),
            pitch_deg=float(row["pitch_deg"]),
            roll_deg=float(row["roll_deg"]),
            cte=float(row["cte"]),
            speed=float(row["speed"]),
            forward_vel=float(row["forward_vel"]),
            steering=float(row["steering"]),
            throttle=float(row["throttle"]),
            brake=float(row["brake"]),
            hit=row["hit"],
        )
        for row in raw_trace_rows
    ]
    centerline = [
        CenterlineWaypoint(
            waypoint_index=int(row["waypoint_index"]),
            distance_m=float(row["distance_m"]),
            pos_x=float(row["pos_x"]),
            pos_y=float(row["pos_y"]),
            pos_z=float(row["pos_z"]),
            heading_x=float(row["heading_x"]),
            heading_z=float(row["heading_z"]),
            yaw_deg=float(row["yaw_deg"]),
            curvature_radpm=float(row["curvature_radpm"]),
            delta_heading_2m_deg=float(row["delta_heading_2m_deg"]),
            curvature_score=float(row["curvature_score"]),
            reference_speed_mps=float(row["reference_speed_mps"]),
        )
        for row in centerline_rows
    ]

    return TrackMapArtifact(
        map_id=str(metadata["map_id"]),
        track_name=str(metadata["track_name"]),
        geometry_hash=str(metadata["geometry_hash"]),
        is_closed=bool(metadata.get("is_closed", True)),
        spacing_m=float(metadata["spacing_m"]),
        total_length_m=float(metadata["total_length_m"]),
        source_lap_count=int(metadata["source_lap_count"]),
        raw_trace=raw_trace,
        centerline=centerline,
    )
