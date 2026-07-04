#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2: Build a dry baseline DEM from offline LiDAR rosbag data."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.dem.visualize_dem import save_dem_heatmap
from src.sensors.rosbag_reader import read_pointcloud_xyz


def load_config(config_path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(f"缺少 PyYAML，无法读取配置: {exc}") from exc

    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    required = ["lidar_topic", "grid_size", "max_frames", "dem_roi"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"配置文件缺少字段: {missing}")
    return config


def filter_roi(points: np.ndarray, roi: dict[str, float]) -> np.ndarray:
    mask = (
        (points[:, 0] >= float(roi["x_min"]))
        & (points[:, 0] <= float(roi["x_max"]))
        & (points[:, 1] >= float(roi["y_min"]))
        & (points[:, 1] <= float(roi["y_max"]))
        & (points[:, 2] >= float(roi["z_min"]))
        & (points[:, 2] <= float(roi["z_max"]))
    )
    return points[mask]


def grid_shape(roi: dict[str, float], grid_size: float) -> tuple[int, int, float, float]:
    width = float(roi["x_max"]) - float(roi["x_min"])
    height = float(roi["y_max"]) - float(roi["y_min"])
    if width <= 0 or height <= 0:
        raise ValueError(f"DEM ROI 范围异常: {roi}")
    nx = int(math.ceil(width / grid_size))
    ny = int(math.ceil(height / grid_size))
    x_max_aligned = float(roi["x_min"]) + nx * grid_size
    y_max_aligned = float(roi["y_min"]) + ny * grid_size
    return nx, ny, x_max_aligned, y_max_aligned


def build_dem_grid(
    points: np.ndarray,
    roi: dict[str, float],
    grid_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    nx, ny, x_max_aligned, y_max_aligned = grid_shape(roi, grid_size)
    x_min = float(roi["x_min"])
    y_min = float(roi["y_min"])

    ix = np.floor((points[:, 0] - x_min) / grid_size).astype(np.int32)
    iy = np.floor((points[:, 1] - y_min) / grid_size).astype(np.int32)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = ix[valid]
    iy = iy[valid]
    z = points[valid, 2]
    if z.size == 0:
        raise ValueError("ROI 内点云没有落入 DEM 栅格")

    flat = (iy.astype(np.int64) * nx + ix.astype(np.int64))
    order = np.lexsort((z, flat))
    flat_sorted = flat[order]
    z_sorted = z[order]

    starts = np.r_[0, np.flatnonzero(np.diff(flat_sorted)) + 1]
    ends = np.r_[starts[1:], flat_sorted.size]
    counts = ends - starts
    lower = starts + (counts - 1) // 2
    upper = starts + counts // 2
    medians = (z_sorted[lower] + z_sorted[upper]) / 2.0

    dem = np.full((ny, nx), np.nan, dtype=np.float32)
    point_count = np.zeros((ny, nx), dtype=np.int32)
    cell_ids = flat_sorted[starts]
    cell_y = (cell_ids // nx).astype(np.int32)
    cell_x = (cell_ids % nx).astype(np.int32)
    dem[cell_y, cell_x] = medians.astype(np.float32)
    point_count[cell_y, cell_x] = counts.astype(np.int32)
    valid_mask = np.isfinite(dem)

    aligned_roi = dict(roi)
    aligned_roi["x_max"] = float(x_max_aligned)
    aligned_roi["y_max"] = float(y_max_aligned)
    metadata = {
        "grid_size": float(grid_size),
        "dem_shape": [int(ny), int(nx)],
        "dem_roi": aligned_roi,
        "valid_cell_count": int(np.count_nonzero(valid_mask)),
        "total_cell_count": int(nx * ny),
        "roi_point_count": int(points.shape[0]),
        "grid_point_count": int(z.size),
        "z_min": float(np.nanmin(dem)),
        "z_max": float(np.nanmax(dem)),
        "z_median": float(np.nanmedian(dem)),
    }
    return dem, valid_mask, point_count, metadata


def build_dem_from_bag(
    dry_bag: str | Path,
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_config(config_path)
    grid_size = float(config["grid_size"])
    max_frames = int(config["max_frames"])
    roi = config["dem_roi"]

    points, read_stats = read_pointcloud_xyz(
        dry_bag,
        topic_name=str(config["lidar_topic"]),
        max_frames=max_frames,
        log_prefix="[S2][reader]",
    )
    roi_points = filter_roi(points, roi)
    print(f"[S2][dem] ROI points: {roi_points.shape[0]} / {points.shape[0]}")
    if roi_points.size == 0:
        raise RuntimeError("DEM ROI 内没有有效点云，请检查 configs/system_config.yaml")

    dem, valid_mask, point_count, metadata = build_dem_grid(roi_points, roi, grid_size)
    metadata.update(
        {
            "stage": "S2_dry_dem_build",
            "source_bag": str(Path(dry_bag).expanduser()),
            "lidar_topic": str(config["lidar_topic"]),
            "max_frames": max_frames,
            "rosbag_read_stats": read_stats,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": "Dry baseline DEM built from LiDAR PointCloud2 median z per XY grid cell.",
        }
    )

    dem_dir = root / "data" / "dem"
    fig_dir = root / "outputs" / "figures"
    dem_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "dem_baseline": dem_dir / "dem_baseline.npy",
        "dem_valid_mask": dem_dir / "dem_valid_mask.npy",
        "dem_point_count": dem_dir / "dem_point_count.npy",
        "dem_metadata": dem_dir / "dem_metadata.json",
        "dem_heatmap": fig_dir / "dem_baseline_heatmap.png",
    }
    np.save(paths["dem_baseline"], dem)
    np.save(paths["dem_valid_mask"], valid_mask)
    np.save(paths["dem_point_count"], point_count)
    metadata["output_files"] = {key: str(path) for key, path in paths.items()}
    with paths["dem_metadata"].open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")

    save_dem_heatmap(
        dem,
        valid_mask,
        paths["dem_heatmap"],
        metadata,
        title="S2 Dry Baseline DEM (z median)",
    )

    print("[S2][dem] DEM grid size:", metadata["grid_size"])
    print("[S2][dem] DEM shape:", metadata["dem_shape"])
    print("[S2][dem] valid cell count:", metadata["valid_cell_count"])
    print(
        "[S2][dem] z stats: "
        f"z_min={metadata['z_min']:.4f}, "
        f"z_max={metadata['z_max']:.4f}, "
        f"z_median={metadata['z_median']:.4f}"
    )
    print("[S2][dem] output files:")
    for path in paths.values():
        print(f"  - {path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build S2 dry baseline DEM from a ROS2 LiDAR bag.")
    parser.add_argument("--dry_bag", required=True, help="Path to dry_baseline_001 rosbag")
    parser.add_argument("--config", required=True, help="Path to configs/system_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    build_dem_from_bag(args.dry_bag, args.config, args.project_root)


if __name__ == "__main__":
    main()
