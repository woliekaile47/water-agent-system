#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2-B: Build a dry ground DEM baseline from offline LiDAR rosbag data."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.dem.build_dem import filter_roi, load_config
from src.sensors.rosbag_reader import read_pointcloud_xyz


def grid_shape(roi: dict[str, float], grid_size: float) -> tuple[int, int, dict[str, float]]:
    width = float(roi["x_max"]) - float(roi["x_min"])
    height = float(roi["y_max"]) - float(roi["y_min"])
    if width <= 0 or height <= 0:
        raise ValueError(f"DEM ROI 范围异常: {roi}")
    nx = int(math.ceil(width / grid_size))
    ny = int(math.ceil(height / grid_size))
    aligned_roi = dict(roi)
    aligned_roi["x_max"] = float(roi["x_min"]) + nx * grid_size
    aligned_roi["y_max"] = float(roi["y_min"]) + ny * grid_size
    return nx, ny, aligned_roi


def build_ground_grid(
    points: np.ndarray,
    roi: dict[str, float],
    grid_size: float,
    percentile: float,
    min_points_per_cell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    nx, ny, aligned_roi = grid_shape(roi, grid_size)
    x_min = float(roi["x_min"])
    y_min = float(roi["y_min"])

    ix = np.floor((points[:, 0] - x_min) / grid_size).astype(np.int32)
    iy = np.floor((points[:, 1] - y_min) / grid_size).astype(np.int32)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = ix[valid]
    iy = iy[valid]
    z = points[valid, 2]
    if z.size == 0:
        raise ValueError("ground DEM ROI 内点云没有落入栅格")

    flat = (iy.astype(np.int64) * nx + ix.astype(np.int64))
    order = np.lexsort((z, flat))
    flat_sorted = flat[order]
    z_sorted = z[order]
    starts = np.r_[0, np.flatnonzero(np.diff(flat_sorted)) + 1]
    ends = np.r_[starts[1:], flat_sorted.size]
    counts = ends - starts
    cell_ids = flat_sorted[starts]
    cell_y = (cell_ids // nx).astype(np.int32)
    cell_x = (cell_ids % nx).astype(np.int32)

    ground_dem = np.full((ny, nx), np.nan, dtype=np.float32)
    z_p10_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    z_p20_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    z_median_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    point_count = np.zeros((ny, nx), dtype=np.int32)

    for start, end, count, cy, cx in zip(starts, ends, counts, cell_y, cell_x):
        point_count[cy, cx] = int(count)
        if count < min_points_per_cell:
            continue
        values = z_sorted[start:end]
        z_p10 = float(np.percentile(values, 10))
        z_p20 = float(np.percentile(values, percentile))
        z_median = float(np.percentile(values, 50))
        z_p10_grid[cy, cx] = z_p10
        z_p20_grid[cy, cx] = z_p20
        z_median_grid[cy, cx] = z_median
        ground_dem[cy, cx] = z_p20

    valid_mask = np.isfinite(ground_dem)
    stats = {
        "grid_point_count": int(z.size),
        "dem_shape": [int(ny), int(nx)],
        "dem_roi": aligned_roi,
        "valid_cell_count": int(np.count_nonzero(valid_mask)),
        "total_cell_count": int(nx * ny),
    }
    grids = {
        "z_p10": z_p10_grid,
        "z_p20": z_p20_grid,
        "z_median": z_median_grid,
    }
    return ground_dem, valid_mask, point_count, grids, stats


def interpolate_nearest(
    dem: np.ndarray,
    valid_mask: np.ndarray,
    max_neighbor_distance_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill invalid cells from the nearest observed cell within a small radius."""
    result = dem.copy()
    fill_mask = np.zeros(valid_mask.shape, dtype=bool)
    valid_positions = np.argwhere(valid_mask)
    if valid_positions.size == 0:
        return result, fill_mask

    radius = max(0, int(max_neighbor_distance_cells))
    if radius <= 0:
        return result, fill_mask

    invalid_positions = np.argwhere(~valid_mask)
    for y, x in invalid_positions:
        y0 = max(0, int(y) - radius)
        y1 = min(dem.shape[0], int(y) + radius + 1)
        x0 = max(0, int(x) - radius)
        x1 = min(dem.shape[1], int(x) + radius + 1)
        local_valid = valid_mask[y0:y1, x0:x1]
        if not np.any(local_valid):
            continue
        local_y, local_x = np.where(local_valid)
        global_y = local_y + y0
        global_x = local_x + x0
        dist2 = (global_y - y) ** 2 + (global_x - x) ** 2
        idx = int(np.argmin(dist2))
        if dist2[idx] <= radius * radius:
            result[y, x] = dem[global_y[idx], global_x[idx]]
            fill_mask[y, x] = True
    return result, fill_mask


def save_heatmap(
    array: np.ndarray,
    output_path: str | Path,
    metadata: dict[str, Any],
    title: str,
    label: str,
    valid_mask: np.ndarray | None = None,
    cmap_name: str = "viridis",
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"缺少 matplotlib，无法生成热力图: {exc}") from exc

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    roi = metadata["dem_roi"]
    data = np.asarray(array)
    if valid_mask is not None:
        data = np.ma.masked_where(~valid_mask, data)
    else:
        data = np.ma.masked_invalid(data)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="lightgray")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    image = ax.imshow(
        data,
        origin="lower",
        extent=[roi["x_min"], roi["x_max"], roi["y_min"], roi["y_max"]],
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(label)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def z_summary(values: np.ndarray) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {
            "z_min": float("nan"),
            "z_max": float("nan"),
            "z_median": float("nan"),
            "z_p10": float("nan"),
            "z_p90": float("nan"),
        }
    return {
        "z_min": float(np.min(valid)),
        "z_max": float(np.max(valid)),
        "z_median": float(np.median(valid)),
        "z_p10": float(np.percentile(valid, 10)),
        "z_p90": float(np.percentile(valid, 90)),
    }


def build_ground_dem_from_bag(
    dry_bag: str | Path,
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_config(config_path)
    ground_config = config.get("ground_dem", {})
    if not ground_config.get("enabled", True):
        raise RuntimeError("configs/system_config.yaml 中 ground_dem.enabled 为 false")

    grid_size = float(config["grid_size"])
    roi = config["dem_roi"]
    max_frames = int(ground_config.get("max_frames", config["max_frames"]))
    ground_z_min = float(ground_config["ground_z_min"])
    ground_z_max = float(ground_config["ground_z_max"])
    percentile = float(ground_config.get("grid_height_percentile", 20))
    min_points_per_cell = int(ground_config.get("min_points_per_cell", 3))

    points, read_stats = read_pointcloud_xyz(
        dry_bag,
        topic_name=str(config["lidar_topic"]),
        max_frames=max_frames,
        log_prefix="[S2-B][reader]",
    )
    roi_points = filter_roi(points, roi)
    ground_mask = (roi_points[:, 2] >= ground_z_min) & (roi_points[:, 2] <= ground_z_max)
    ground_points = roi_points[ground_mask]
    print(f"[S2-B][ground] ROI points: {roi_points.shape[0]} / {points.shape[0]}")
    print(
        "[S2-B][ground] ground z filter: "
        f"[{ground_z_min:.3f}, {ground_z_max:.3f}], points={ground_points.shape[0]}"
    )
    if ground_points.size == 0:
        raise RuntimeError("ground_z_min / ground_z_max 过滤后没有有效地面点")

    ground_dem, valid_mask, point_count, stat_grids, grid_stats = build_ground_grid(
        ground_points,
        roi,
        grid_size,
        percentile,
        min_points_per_cell,
    )
    valid_ratio = (
        grid_stats["valid_cell_count"] / grid_stats["total_cell_count"]
        if grid_stats["total_cell_count"]
        else 0.0
    )

    interpolation_config = ground_config.get("interpolation", {})
    interpolation_enabled = bool(interpolation_config.get("enabled", True))
    interpolation_method = str(interpolation_config.get("method", "nearest"))
    max_neighbor_distance_cells = int(interpolation_config.get("max_neighbor_distance_cells", 3))
    if interpolation_enabled and interpolation_method != "nearest":
        raise ValueError(f"当前只支持 nearest 插值，不支持: {interpolation_method}")
    if interpolation_enabled:
        interpolated, interpolation_fill_mask = interpolate_nearest(
            ground_dem,
            valid_mask,
            max_neighbor_distance_cells,
        )
    else:
        interpolated = ground_dem.copy()
        interpolation_fill_mask = np.zeros(valid_mask.shape, dtype=bool)

    dem_dir = root / "data" / "dem"
    fig_dir = root / "outputs" / "figures"
    dem_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ground_dem": dem_dir / "ground_dem.npy",
        "ground_dem_valid_mask": dem_dir / "ground_dem_valid_mask.npy",
        "ground_dem_point_count": dem_dir / "ground_dem_point_count.npy",
        "ground_dem_interpolated": dem_dir / "ground_dem_interpolated.npy",
        "ground_dem_metadata": dem_dir / "ground_dem_metadata.json",
        "ground_dem_heatmap": fig_dir / "ground_dem_heatmap.png",
        "ground_dem_interpolated_heatmap": fig_dir / "ground_dem_interpolated_heatmap.png",
        "ground_dem_point_count_figure": fig_dir / "ground_dem_point_count.png",
    }

    metadata: dict[str, Any] = {
        "stage": "S2-B_ground_dem_build",
        "method": str(ground_config.get("method", "low_percentile_grid")),
        "grid_size": grid_size,
        "dem_shape": grid_stats["dem_shape"],
        "dem_roi": grid_stats["dem_roi"],
        "ground_z_filter": {
            "ground_z_min": ground_z_min,
            "ground_z_max": ground_z_max,
        },
        "grid_height_percentile": percentile,
        "min_points_per_cell": min_points_per_cell,
        "valid_cell_count": grid_stats["valid_cell_count"],
        "total_cell_count": grid_stats["total_cell_count"],
        "valid_ratio": float(valid_ratio),
        "source_bag": str(Path(dry_bag).expanduser()),
        "lidar_topic": str(config["lidar_topic"]),
        "max_frames": max_frames,
        "rosbag_read_stats": read_stats,
        "roi_point_count": int(roi_points.shape[0]),
        "ground_point_count": int(ground_points.shape[0]),
        "interpolation": {
            "enabled": interpolation_enabled,
            "method": interpolation_method,
            "max_neighbor_distance_cells": max_neighbor_distance_cells,
            "filled_cell_count": int(np.count_nonzero(interpolation_fill_mask)),
            "note": "Interpolated cells are gap-filled estimates, not direct observations.",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata.update(z_summary(ground_dem))

    np.save(paths["ground_dem"], ground_dem)
    np.save(paths["ground_dem_valid_mask"], valid_mask)
    np.save(paths["ground_dem_point_count"], point_count)
    np.save(paths["ground_dem_interpolated"], interpolated)

    metadata["output_files"] = {key: str(path) for key, path in paths.items()}
    with paths["ground_dem_metadata"].open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")

    save_heatmap(
        ground_dem,
        paths["ground_dem_heatmap"],
        metadata,
        title="S2-B Ground DEM (z_p20 observed cells)",
        label="ground elevation z_p20 (m)",
        valid_mask=valid_mask,
        cmap_name="terrain",
    )
    save_heatmap(
        interpolated,
        paths["ground_dem_interpolated_heatmap"],
        metadata,
        title="S2-B Ground DEM (nearest interpolated)",
        label="ground elevation z_p20 / nearest fill (m)",
        valid_mask=np.isfinite(interpolated),
        cmap_name="terrain",
    )
    save_heatmap(
        point_count.astype(np.float32),
        paths["ground_dem_point_count_figure"],
        metadata,
        title="S2-B Ground DEM point count per cell",
        label="ground-filtered point count",
        valid_mask=point_count > 0,
        cmap_name="magma",
    )

    print("[S2-B][ground] ground DEM shape:", metadata["dem_shape"])
    print("[S2-B][ground] valid cell count:", metadata["valid_cell_count"])
    print(f"[S2-B][ground] valid ratio: {metadata['valid_ratio']:.4f}")
    print(
        "[S2-B][ground] z stats: "
        f"z_min={metadata['z_min']:.4f}, "
        f"z_max={metadata['z_max']:.4f}, "
        f"z_median={metadata['z_median']:.4f}"
    )
    if metadata["valid_cell_count"] < 50 or metadata["valid_ratio"] < 0.03:
        print(
            "[S2-B][WARN] ground DEM 有效栅格较少，建议检查 ground_z_min / "
            "ground_z_max 或 dem_roi。"
        )
    print("[S2-B][ground] output files:")
    for path in paths.values():
        print(f"  - {path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build S2-B ground DEM from a dry LiDAR rosbag.")
    parser.add_argument("--dry_bag", required=True, help="Path to dry_baseline_001 rosbag")
    parser.add_argument("--config", required=True, help="Path to configs/system_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    build_ground_dem_from_bag(args.dry_bag, args.config, args.project_root)


if __name__ == "__main__":
    main()
