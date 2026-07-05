#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a scene-specific ground DEM from an offline LiDAR rosbag."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dem.build_surface_dem_from_rosbag import (
    find_rosbag_dir,
    get_case_config,
    get_scene_config,
    load_yaml,
    resolve_project_path,
    save_heatmap,
)
from src.sensors.rosbag_reader import load_ros_modules, open_sequential_reader, pointcloud2_to_xyz


def read_pointcloud_frames(
    bag_path: Path,
    topic_name: str,
    max_frames: int,
    frame_stride: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rosbag2_py, deserialize_message, get_message, point_cloud2 = load_ros_modules()
    _ = rosbag2_py
    reader = open_sequential_reader(bag_path)
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topic_name not in topic_types:
        available = ", ".join(sorted(topic_types)) or "(none)"
        raise RuntimeError(f"Topic does not exist: {topic_name}. Available topics: {available}")
    if topic_types[topic_name] != "sensor_msgs/msg/PointCloud2":
        raise RuntimeError(f"Topic {topic_name} is {topic_types[topic_name]}, not PointCloud2.")

    msg_type = get_message(topic_types[topic_name])
    max_frames = max(1, int(max_frames))
    frame_stride = max(1, int(frame_stride))
    topic_frames_seen = 0
    frame_count = 0
    total_points = 0
    chunks: list[np.ndarray] = []

    print(f"[ground_dem] open rosbag: {bag_path}")
    print(f"[ground_dem] topic: {topic_name}, max_frames={max_frames}, frame_stride={frame_stride}")
    while reader.has_next() and frame_count < max_frames:
        topic, data, timestamp = reader.read_next()
        if topic != topic_name:
            continue
        topic_frames_seen += 1
        if (topic_frames_seen - 1) % frame_stride != 0:
            continue
        msg = deserialize_message(data, msg_type)
        points = pointcloud2_to_xyz(msg, point_cloud2)
        frame_count += 1
        total_points += int(points.shape[0])
        if points.size:
            chunks.append(points.astype(np.float32, copy=False))
        print(f"[ground_dem] frame={frame_count:03d}, timestamp={timestamp}, points={points.shape[0]}")

    if frame_count == 0:
        raise RuntimeError(f"No PointCloud2 frames were read from topic {topic_name}.")
    if not chunks:
        raise RuntimeError("No valid xyz points were read from PointCloud2 frames.")

    return np.concatenate(chunks, axis=0), {
        "frame_count": int(frame_count),
        "topic_frames_seen": int(topic_frames_seen),
        "total_points": int(total_points),
    }


def merge_ground_config(config: dict[str, Any], scene_type: str | None) -> dict[str, Any]:
    base = dict(config.get("ground_dem", {}))
    scene_config = get_scene_config(config, scene_type)
    scene_ground = dict(scene_config.get("ground_dem", {}))
    merged = {**base, **scene_ground}
    if "roi" in base or "roi" in scene_ground:
        merged["roi"] = {**dict(base.get("roi", {})), **dict(scene_ground.get("roi", {}))}
    if "interpolation" in base or "interpolation" in scene_ground:
        merged["interpolation"] = {
            **dict(base.get("interpolation", {})),
            **dict(scene_ground.get("interpolation", {})),
        }
    return merged


def filter_and_make_roi(points: np.ndarray, ground_config: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    roi_config = dict(ground_config.get("roi", {}))
    z_min = float(roi_config.get("z_min_m", -np.inf))
    z_max = float(roi_config.get("z_max_m", np.inf))
    filtered = points[np.isfinite(points).all(axis=1)]
    filtered = filtered[(filtered[:, 2] >= z_min) & (filtered[:, 2] <= z_max)]
    if filtered.size == 0:
        raise RuntimeError("No points remain after z filtering. Refuse to build fake ground DEM.")

    grid_size = float(ground_config.get("grid_size", 0.10))
    mode = str(roi_config.get("mode", "auto_from_points"))
    if mode == "manual":
        required = ("x_min", "x_max", "y_min", "y_max")
        missing = [key for key in required if key not in roi_config]
        if missing:
            raise ValueError(f"Manual ROI is missing fields: {missing}")
        roi = {key: float(roi_config[key]) for key in required}
        in_roi = (
            (filtered[:, 0] >= roi["x_min"]) & (filtered[:, 0] < roi["x_max"]) &
            (filtered[:, 1] >= roi["y_min"]) & (filtered[:, 1] < roi["y_max"])
        )
        filtered = filtered[in_roi]
    elif mode == "auto_from_points":
        margin = float(roi_config.get("margin_m", 0.0))
        x_min = np.floor((float(np.min(filtered[:, 0])) - margin) / grid_size) * grid_size
        x_max = np.ceil((float(np.max(filtered[:, 0])) + margin) / grid_size) * grid_size
        y_min = np.floor((float(np.min(filtered[:, 1])) - margin) / grid_size) * grid_size
        y_max = np.ceil((float(np.max(filtered[:, 1])) + margin) / grid_size) * grid_size
        roi = {"x_min": float(x_min), "x_max": float(x_max), "y_min": float(y_min), "y_max": float(y_max)}
    else:
        raise ValueError(f"Unsupported ground DEM ROI mode: {mode}")

    if filtered.size == 0:
        raise RuntimeError("No points remain inside configured ROI. Refuse to build fake ground DEM.")
    roi["z_min"] = z_min
    roi["z_max"] = z_max
    return filtered, roi


def rasterize_ground(
    points: np.ndarray,
    roi: dict[str, float],
    grid_size: float,
    z_statistic: str,
    min_points_per_cell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx = int(np.ceil((roi["x_max"] - roi["x_min"]) / grid_size))
    ny = int(np.ceil((roi["y_max"] - roi["y_min"]) / grid_size))
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Invalid DEM shape from ROI/grid: ny={ny}, nx={nx}")

    ix = np.floor((points[:, 0] - roi["x_min"]) / grid_size).astype(np.int64)
    iy = np.floor((points[:, 1] - roi["y_min"]) / grid_size).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not np.any(inside):
        raise RuntimeError("No filtered points fall inside the DEM grid.")
    ix = ix[inside]
    iy = iy[inside]
    z = points[inside, 2].astype(np.float32)
    flat = iy * nx + ix
    order = np.lexsort((z, flat))
    flat_sorted = flat[order]
    z_sorted = z[order]
    starts = np.r_[0, np.flatnonzero(np.diff(flat_sorted)) + 1]
    ends = np.r_[starts[1:], flat_sorted.size]

    ground_dem = np.full((ny, nx), np.nan, dtype=np.float32)
    point_count = np.zeros((ny, nx), dtype=np.int32)
    min_points = max(1, int(min_points_per_cell))
    percentile = 20.0
    if z_statistic.startswith("p"):
        percentile = float(z_statistic[1:])
    elif z_statistic != "median":
        raise ValueError(f"Unsupported ground z_statistic: {z_statistic}")
    elif z_statistic == "median":
        percentile = 50.0

    for start, end in zip(starts, ends):
        count = int(end - start)
        cell_id = int(flat_sorted[start])
        cy = cell_id // nx
        cx = cell_id % nx
        point_count[cy, cx] = count
        if count >= min_points:
            ground_dem[cy, cx] = np.float32(np.percentile(z_sorted[start:end], percentile))

    valid_mask = np.isfinite(ground_dem)
    if not np.any(valid_mask):
        raise RuntimeError("Ground DEM has zero valid cells. Refuse to generate fake output.")
    return ground_dem, valid_mask, point_count


def interpolate_nearest(dem: np.ndarray, valid_mask: np.ndarray, max_distance_cells: int) -> np.ndarray:
    interpolated = dem.copy()
    valid_positions = np.argwhere(valid_mask)
    if valid_positions.size == 0:
        return interpolated
    max_distance = max(0, int(max_distance_cells))
    for y, x in np.argwhere(~valid_mask):
        y0 = max(0, y - max_distance)
        y1 = min(valid_mask.shape[0], y + max_distance + 1)
        x0 = max(0, x - max_distance)
        x1 = min(valid_mask.shape[1], x + max_distance + 1)
        local_valid = valid_mask[y0:y1, x0:x1]
        if not np.any(local_valid):
            continue
        local_y, local_x = np.where(local_valid)
        gy = local_y + y0
        gx = local_x + x0
        dist2 = (gy - y) ** 2 + (gx - x) ** 2
        idx = int(np.argmin(dist2))
        interpolated[y, x] = dem[gy[idx], gx[idx]]
    return interpolated


def build_ground_dem_from_rosbag(
    config_path: str | Path,
    project_root: str | Path,
    case_name: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)
    selected_case, case_config = get_case_config(config, case_name)
    if case_config.get("data_type") != "dry_baseline":
        raise ValueError(f"Case {selected_case} is not a dry_baseline case.")

    scene_type = str(case_config.get("scene_type", selected_case))
    scene_config = get_scene_config(config, scene_type)
    ground_config = merge_ground_config(config, scene_type)
    output_dir = resolve_project_path(
        root,
        scene_config.get("ground_dem_output_dir", f"data/dem/{scene_type}"),
    )
    figure_dir = resolve_project_path(root, config["output"]["figure_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    bag = find_rosbag_dir(config["data_root"], str(case_config["bag_search_keyword"]))
    points, read_stats = read_pointcloud_frames(
        bag,
        str(config["lidar"]["topic"]),
        int(ground_config.get("max_frames", config["lidar"].get("max_frames", 80))),
        int(ground_config.get("frame_stride", config["lidar"].get("frame_stride", 1))),
    )
    filtered_points, roi = filter_and_make_roi(points, ground_config)
    grid_size = float(ground_config.get("grid_size", 0.10))
    ground_dem, valid_mask, point_count = rasterize_ground(
        filtered_points,
        roi,
        grid_size,
        str(ground_config.get("z_statistic", "p20")),
        int(ground_config.get("min_points_per_cell", 3)),
    )

    interpolation_config = dict(ground_config.get("interpolation", {}))
    if bool(interpolation_config.get("enabled", True)):
        interpolated = interpolate_nearest(
            ground_dem,
            valid_mask,
            int(interpolation_config.get("max_neighbor_distance_cells", 3)),
        )
    else:
        interpolated = ground_dem.copy()

    paths = {
        "ground_dem": output_dir / "ground_dem.npy",
        "ground_dem_valid_mask": output_dir / "ground_dem_valid_mask.npy",
        "ground_dem_point_count": output_dir / "ground_dem_point_count.npy",
        "ground_dem_interpolated": output_dir / "ground_dem_interpolated.npy",
        "ground_dem_metadata": output_dir / "ground_dem_metadata.json",
        "ground_dem_heatmap": figure_dir / f"ground_dem_heatmap_{scene_type}.png",
        "ground_dem_interpolated_heatmap": figure_dir / f"ground_dem_interpolated_heatmap_{scene_type}.png",
        "ground_dem_point_count_figure": figure_dir / f"ground_dem_point_count_{scene_type}.png",
    }
    np.save(paths["ground_dem"], ground_dem)
    np.save(paths["ground_dem_valid_mask"], valid_mask)
    np.save(paths["ground_dem_point_count"], point_count)
    np.save(paths["ground_dem_interpolated"], interpolated)
    save_heatmap(
        ground_dem,
        valid_mask,
        paths["ground_dem_heatmap"],
        f"Scene Ground DEM - {scene_type}",
        "ground z (m)",
    )
    save_heatmap(
        interpolated,
        np.isfinite(interpolated),
        paths["ground_dem_interpolated_heatmap"],
        f"Scene Ground DEM Interpolated - {scene_type}",
        "ground z (m)",
    )
    save_heatmap(
        point_count.astype(np.float32),
        point_count > 0,
        paths["ground_dem_point_count_figure"],
        f"Scene Ground DEM Point Count - {scene_type}",
        "points / cell",
        cmap="magma",
    )

    metadata = {
        "stage": "S4_real_scene_ground_dem_build",
        "source_bag": str(bag),
        "case_name": selected_case,
        "scene_type": scene_type,
        "lidar_topic": str(config["lidar"]["topic"]),
        "frame_count": read_stats["frame_count"],
        "topic_frames_seen": read_stats["topic_frames_seen"],
        "total_points": read_stats["total_points"],
        "filtered_points": int(filtered_points.shape[0]),
        "valid_cell_count": int(np.count_nonzero(valid_mask)),
        "total_cell_count": int(valid_mask.size),
        "valid_ratio": float(np.count_nonzero(valid_mask) / max(1, valid_mask.size)),
        "grid_size": grid_size,
        "grid_resolution": grid_size,
        "dem_shape": [int(ground_dem.shape[0]), int(ground_dem.shape[1])],
        "dem_roi": roi,
        "roi_settings": roi,
        "ground_config": ground_config,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_files": {key: str(value) for key, value in paths.items()},
    }
    with paths["ground_dem_metadata"].open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[ground_dem] case_name: {selected_case}")
    print(f"[ground_dem] scene_type: {scene_type}")
    print(f"[ground_dem] source_bag: {bag}")
    print(f"[ground_dem] frame_count: {metadata['frame_count']}")
    print(f"[ground_dem] total_points: {metadata['total_points']}")
    print(f"[ground_dem] filtered_points: {metadata['filtered_points']}")
    print(f"[ground_dem] valid_cell_count: {metadata['valid_cell_count']}")
    print("[ground_dem] output paths:")
    for path in paths.values():
        print(f"  - {path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scene-specific ground DEM from offline LiDAR rosbag.")
    parser.add_argument("--config", required=True, help="Path to configs/surface_dem_config.yaml")
    parser.add_argument("--case", required=True, help="Dry baseline case name")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    build_ground_dem_from_rosbag(args.config, args.project_root, args.case)


if __name__ == "__main__":
    main()
