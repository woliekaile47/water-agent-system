#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4-real: build current surface DEM from an offline LiDAR rosbag."""

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

from src.sensors.rosbag_reader import load_ros_modules, open_sequential_reader, pointcloud2_to_xyz


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(f"PyYAML is required to read config: {exc}") from exc

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "surface_dem" not in data:
        raise ValueError("Config must contain a top-level 'surface_dem' field.")
    return data["surface_dem"]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def get_case_config(config: dict[str, Any], case_name: str | None) -> tuple[str, dict[str, Any]]:
    cases = config.get("cases", {})
    selected = case_name or config.get("active_case")
    if not selected:
        raise ValueError("No case specified and surface_dem.active_case is empty.")
    if selected not in cases:
        available = ", ".join(sorted(cases)) or "(none)"
        raise ValueError(f"Unknown case: {selected}. Available cases: {available}")
    return str(selected), dict(cases[selected])


def find_rosbag_dir(data_root: str | Path, keyword: str) -> Path:
    root = Path(data_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")

    metadata_hits = [
        path.parent
        for path in root.rglob("metadata.yaml")
        if keyword in str(path.parent)
    ]
    if metadata_hits:
        return sorted(metadata_hits, key=lambda p: (len(str(p)), str(p)))[0]

    dir_hits = [
        path
        for path in root.rglob("*")
        if path.is_dir() and keyword in path.name
    ]
    if dir_hits:
        return sorted(dir_hits, key=lambda p: (len(str(p)), str(p)))[0]

    raise FileNotFoundError(
        f"Could not find rosbag directory containing keyword '{keyword}' under {root}. "
        "Try: find ~/water_agent_data -maxdepth 4 -type f -name \"metadata.yaml\""
    )


def parse_dem_grid(metadata: dict[str, Any], ground_dem_shape: tuple[int, int]) -> dict[str, Any]:
    grid_size = (
        metadata.get("grid_size")
        or metadata.get("resolution")
        or metadata.get("cell_size")
        or metadata.get("cell_size_m")
    )
    if grid_size is None:
        raise ValueError("ground_dem_metadata is missing grid_size/resolution/cell_size/cell_size_m.")
    grid_size = float(grid_size)

    dem_shape = metadata.get("dem_shape") or list(ground_dem_shape)
    if len(dem_shape) != 2:
        raise ValueError(f"Invalid dem_shape in metadata: {dem_shape}")
    ny, nx = int(dem_shape[0]), int(dem_shape[1])
    if (ny, nx) != tuple(ground_dem_shape):
        raise ValueError(
            f"ground_dem shape {ground_dem_shape} does not match metadata dem_shape {(ny, nx)}"
        )

    roi = metadata.get("dem_roi") or metadata.get("bounds") or metadata.get("roi") or {}
    x_min = roi.get("x_min", metadata.get("x_min"))
    x_max = roi.get("x_max", metadata.get("x_max"))
    y_min = roi.get("y_min", metadata.get("y_min"))
    y_max = roi.get("y_max", metadata.get("y_max"))
    if x_min is None or y_min is None:
        raise ValueError("ground_dem_metadata must provide x_min/y_min in dem_roi, bounds, roi, or top-level fields.")
    x_min = float(x_min)
    y_min = float(y_min)
    x_max = float(x_max) if x_max is not None else x_min + nx * grid_size
    y_max = float(y_max) if y_max is not None else y_min + ny * grid_size
    return {
        "grid_size_m": grid_size,
        "dem_shape": [ny, nx],
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }


def read_filtered_points(
    bag_path: Path,
    topic_name: str,
    max_frames: int,
    frame_stride: int,
    grid: dict[str, Any],
    water_region_mask: np.ndarray,
    filtering: dict[str, Any],
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
    z_min = float(filtering.get("z_min_m", -np.inf))
    z_max = float(filtering.get("z_max_m", np.inf))
    use_dem_bounds = bool(filtering.get("use_dem_bounds", True))
    use_water_region_mask = bool(filtering.get("use_water_region_mask", True))
    ny, nx = int(grid["dem_shape"][0]), int(grid["dem_shape"][1])
    grid_size = float(grid["grid_size_m"])

    chunks: list[np.ndarray] = []
    topic_frames_seen = 0
    frames_read = 0
    points_total = 0
    points_after_filter = 0

    print(f"[surface_dem] open rosbag: {bag_path}")
    print(f"[surface_dem] topic: {topic_name}, max_frames={max_frames}, frame_stride={frame_stride}")
    while reader.has_next() and frames_read < max_frames:
        topic, data, timestamp = reader.read_next()
        if topic != topic_name:
            continue
        topic_frames_seen += 1
        if (topic_frames_seen - 1) % frame_stride != 0:
            continue

        msg = deserialize_message(data, msg_type)
        points = pointcloud2_to_xyz(msg, point_cloud2)
        frames_read += 1
        points_total += int(points.shape[0])

        if points.size:
            mask = np.isfinite(points).all(axis=1)
            mask &= (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
            if use_dem_bounds:
                mask &= (points[:, 0] >= grid["x_min"]) & (points[:, 0] < grid["x_max"])
                mask &= (points[:, 1] >= grid["y_min"]) & (points[:, 1] < grid["y_max"])

            filtered = points[mask]
            if filtered.size and use_water_region_mask:
                ix = np.floor((filtered[:, 0] - grid["x_min"]) / grid_size).astype(np.int64)
                iy = np.floor((filtered[:, 1] - grid["y_min"]) / grid_size).astype(np.int64)
                inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
                region = np.zeros_like(inside, dtype=bool)
                region[inside] = water_region_mask[iy[inside], ix[inside]]
                filtered = filtered[region]
            if filtered.size:
                chunks.append(filtered.astype(np.float32, copy=False))
                points_after_filter += int(filtered.shape[0])

        print(
            "[surface_dem] "
            f"frame={frames_read:03d}, timestamp={timestamp}, "
            f"points={points.shape[0]}, kept_total={points_after_filter}"
        )

    if frames_read == 0:
        raise RuntimeError(f"No PointCloud2 frames were read from topic {topic_name}.")
    if not chunks:
        raise RuntimeError(
            "No valid LiDAR points remained after DEM/water-region/z filtering. "
            "Refuse to generate fake surface DEM."
        )

    xyz = np.concatenate(chunks, axis=0)
    stats = {
        "frames_read": int(frames_read),
        "topic_frames_seen": int(topic_frames_seen),
        "points_total": int(points_total),
        "points_after_filter": int(points_after_filter),
    }
    return xyz, stats


def rasterize_surface(
    points: np.ndarray,
    grid: dict[str, Any],
    min_points_per_cell: int,
    z_statistic: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if z_statistic != "median":
        raise ValueError(f"Unsupported z_statistic: {z_statistic}. Only 'median' is implemented.")

    ny, nx = int(grid["dem_shape"][0]), int(grid["dem_shape"][1])
    grid_size = float(grid["grid_size_m"])
    ix = np.floor((points[:, 0] - grid["x_min"]) / grid_size).astype(np.int64)
    iy = np.floor((points[:, 1] - grid["y_min"]) / grid_size).astype(np.int64)
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

    surface_dem = np.full((ny, nx), np.nan, dtype=np.float32)
    point_count = np.zeros((ny, nx), dtype=np.int32)
    min_points = max(1, int(min_points_per_cell))
    for start, end in zip(starts, ends):
        count = int(end - start)
        cell_id = int(flat_sorted[start])
        cy = cell_id // nx
        cx = cell_id % nx
        point_count[cy, cx] = count
        if count >= min_points:
            surface_dem[cy, cx] = np.float32(np.median(z_sorted[start:end]))

    valid_mask = np.isfinite(surface_dem)
    return surface_dem, valid_mask, point_count


def save_heatmap(
    array: np.ndarray,
    valid_mask: np.ndarray,
    output_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate figure: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ma.masked_where(~valid_mask, array)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    image = ax.imshow(data, origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_surface_dem(
    config_path: str | Path,
    project_root: str | Path,
    case_name: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)
    selected_case, case_config = get_case_config(config, case_name)
    output_name = str(case_config.get("output_name", selected_case))
    bag = find_rosbag_dir(config["data_root"], str(case_config["bag_search_keyword"]))

    ground_dem_path = resolve_project_path(root, config["dem"]["ground_dem_interpolated_path"])
    ground_mask_path = resolve_project_path(root, config["dem"]["ground_dem_valid_mask_path"])
    metadata_path = resolve_project_path(root, config["dem"]["ground_dem_metadata_path"])
    water_mask_path = resolve_project_path(root, config["fusion"]["water_region_mask_path"])
    ground_dem = np.load(ground_dem_path)
    ground_valid_mask = np.load(ground_mask_path).astype(bool)
    dem_metadata = load_json(metadata_path)
    water_region_mask = np.load(water_mask_path).astype(bool)
    if water_region_mask.shape != ground_dem.shape or ground_valid_mask.shape != ground_dem.shape:
        raise ValueError(
            "ground DEM, ground valid mask, and water region mask shapes must match: "
            f"{ground_dem.shape}, {ground_valid_mask.shape}, {water_region_mask.shape}"
        )

    grid = parse_dem_grid(dem_metadata, tuple(ground_dem.shape))
    points, read_stats = read_filtered_points(
        bag_path=bag,
        topic_name=str(config["lidar"]["topic"]),
        max_frames=int(config["lidar"].get("max_frames", 30)),
        frame_stride=int(config["lidar"].get("frame_stride", 1)),
        grid=grid,
        water_region_mask=water_region_mask,
        filtering=config.get("filtering", {}),
    )
    surface_dem, surface_valid_mask, point_count = rasterize_surface(
        points=points,
        grid=grid,
        min_points_per_cell=int(config["rasterization"].get("min_points_per_cell", 1)),
        z_statistic=str(config["rasterization"].get("z_statistic", "median")),
    )

    if bool(config.get("filtering", {}).get("use_water_region_mask", True)):
        surface_valid_mask &= water_region_mask
        surface_dem = np.where(surface_valid_mask, surface_dem, np.nan).astype(np.float32)
        point_count = np.where(water_region_mask, point_count, 0).astype(np.int32)

    water_region_cell_count = int(np.count_nonzero(water_region_mask))
    valid_surface_cell_count = int(np.count_nonzero(surface_valid_mask & water_region_mask))
    valid_ratio = (
        float(valid_surface_cell_count / water_region_cell_count)
        if water_region_cell_count > 0
        else 0.0
    )
    warnings: list[str] = []
    if water_region_cell_count == 0:
        raise RuntimeError("water_region_mask has no valid cells.")
    if valid_surface_cell_count == 0:
        raise RuntimeError("Surface DEM has zero valid cells in water_region_mask. Refuse fake output.")
    if valid_ratio < 0.05:
        warnings.append("low_data_quality: surface_valid_ratio_in_water_region is below 0.05")

    surface_dir = resolve_project_path(root, config["output"]["surface_dem_dir"]) / output_name
    figure_dir = resolve_project_path(root, config["output"]["figure_dir"])
    surface_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "surface_dem": surface_dir / "surface_dem.npy",
        "surface_dem_valid_mask": surface_dir / "surface_dem_valid_mask.npy",
        "surface_dem_point_count": surface_dir / "surface_dem_point_count.npy",
        "surface_dem_metadata": surface_dir / "surface_dem_metadata.json",
        "surface_dem_heatmap": figure_dir / f"surface_dem_heatmap_{output_name}.png",
        "surface_dem_point_count_figure": figure_dir / f"surface_dem_point_count_{output_name}.png",
    }

    np.save(paths["surface_dem"], surface_dem)
    np.save(paths["surface_dem_valid_mask"], surface_valid_mask)
    np.save(paths["surface_dem_point_count"], point_count)
    save_heatmap(
        surface_dem,
        surface_valid_mask,
        paths["surface_dem_heatmap"],
        f"S4-real Surface DEM - {output_name}",
        "z median (m)",
    )
    save_heatmap(
        point_count.astype(np.float32),
        point_count > 0,
        paths["surface_dem_point_count_figure"],
        f"S4-real Surface DEM Point Count - {output_name}",
        "points / cell",
        cmap="magma",
    )

    metadata = {
        "stage": "S4_real_surface_dem_build",
        "case_name": output_name,
        "rosbag_path": str(bag),
        "lidar_topic": str(config["lidar"]["topic"]),
        "frames_read": read_stats["frames_read"],
        "topic_frames_seen": read_stats["topic_frames_seen"],
        "points_total": read_stats["points_total"],
        "points_after_filter": read_stats["points_after_filter"],
        "dem_shape": grid["dem_shape"],
        "grid_size_m": grid["grid_size_m"],
        "dem_bounds": {
            "x_min": grid["x_min"],
            "x_max": grid["x_max"],
            "y_min": grid["y_min"],
            "y_max": grid["y_max"],
        },
        "valid_surface_cell_count": valid_surface_cell_count,
        "water_region_cell_count": water_region_cell_count,
        "surface_valid_ratio_in_water_region": valid_ratio,
        "z_statistic": str(config["rasterization"].get("z_statistic", "median")),
        "known_depth_cm": case_config.get("known_depth_cm"),
        "warning": warnings,
        "note": config.get("note"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_files": {key: str(value) for key, value in paths.items()},
    }
    with paths["surface_dem_metadata"].open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[surface_dem] case_name: {output_name}")
    print(f"[surface_dem] rosbag_path: {bag}")
    print(f"[surface_dem] frames_read: {metadata['frames_read']}")
    print(f"[surface_dem] points_after_filter: {metadata['points_after_filter']}")
    print(f"[surface_dem] valid_surface_cell_count: {valid_surface_cell_count}")
    print(f"[surface_dem] surface_valid_ratio_in_water_region: {valid_ratio:.4f}")
    if warnings:
        print(f"[surface_dem][WARN] {'; '.join(warnings)}")
    print("[surface_dem] output paths:")
    for path in paths.values():
        print(f"  - {path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build S4-real surface DEM from offline LiDAR rosbag.")
    parser.add_argument("--config", required=True, help="Path to configs/surface_dem_config.yaml")
    parser.add_argument("--case", help="Case name, e.g. water_sim_13cm_001")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    build_surface_dem(args.config, args.project_root, args.case)


if __name__ == "__main__":
    main()
