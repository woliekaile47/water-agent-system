#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4-real: invert water depth from surface DEM minus ground DEM."""

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

from src.dem.build_surface_dem_from_rosbag import get_case_config, load_yaml, resolve_project_path


NOTE = (
    "This depth map is computed from offline LiDAR surface DEM minus ground DEM. "
    "It does not use configured_depth. Accuracy depends on calibration, ROI mapping, "
    "point density, and water surface returns."
)


def save_depth_heatmap(depth_map_m: np.ndarray, valid_mask: np.ndarray, output_path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate surface depth heatmap: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth_cm = depth_map_m.astype(np.float64) * 100.0
    masked = np.ma.masked_where(~valid_mask, depth_cm)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    image = ax.imshow(masked, origin="lower", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("depth (cm)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _stats_cm(depth_map_m: np.ndarray, valid_mask: np.ndarray) -> dict[str, float | None]:
    if not np.any(valid_mask):
        return {
            "max_depth_cm": None,
            "mean_depth_cm": None,
            "median_depth_cm": None,
            "min_depth_cm": None,
        }
    values = depth_map_m[valid_mask].astype(np.float64) * 100.0
    return {
        "max_depth_cm": float(np.max(values)),
        "mean_depth_cm": float(np.mean(values)),
        "median_depth_cm": float(np.median(values)),
        "min_depth_cm": float(np.min(values)),
    }


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.2f}"


def invert_surface_depth(
    config_path: str | Path,
    project_root: str | Path,
    case_name: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)
    selected_case, case_config = get_case_config(config, case_name)
    output_name = str(case_config.get("output_name", selected_case))

    surface_dir = resolve_project_path(root, config["output"]["surface_dem_dir"]) / output_name
    hydrology_dir = resolve_project_path(root, config["output"]["hydrology_dir"]) / output_name
    json_dir = resolve_project_path(root, config["output"]["json_dir"])
    figure_dir = resolve_project_path(root, config["output"]["figure_dir"])
    hydrology_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    surface_dem_path = surface_dir / "surface_dem.npy"
    surface_valid_path = surface_dir / "surface_dem_valid_mask.npy"
    ground_dem_path = resolve_project_path(root, config["dem"]["ground_dem_interpolated_path"])
    ground_valid_path = resolve_project_path(root, config["dem"]["ground_dem_valid_mask_path"])
    water_mask_path = resolve_project_path(root, config["fusion"]["water_region_mask_path"])

    surface_dem = np.load(surface_dem_path)
    surface_valid_mask = np.load(surface_valid_path).astype(bool)
    ground_dem = np.load(ground_dem_path)
    ground_valid_mask = np.load(ground_valid_path).astype(bool)
    water_region_mask = np.load(water_mask_path).astype(bool)

    if not (surface_dem.shape == surface_valid_mask.shape == ground_dem.shape == ground_valid_mask.shape == water_region_mask.shape):
        raise ValueError(
            "Shape mismatch among surface DEM, ground DEM, and masks: "
            f"{surface_dem.shape}, {surface_valid_mask.shape}, {ground_dem.shape}, "
            f"{ground_valid_mask.shape}, {water_region_mask.shape}"
        )

    depth_config = config.get("depth", {})
    min_valid_depth_m = float(depth_config.get("min_valid_depth_m", 0.005))
    max_valid_depth_m = float(depth_config.get("max_valid_depth_m", 1.5))
    negative_policy = str(depth_config.get("negative_depth_policy", "set_zero"))
    outside_value = float(depth_config.get("outside_mask_value", 0.0))

    base_valid_mask = surface_valid_mask & ground_valid_mask & water_region_mask
    raw_depth_m = surface_dem.astype(np.float32) - ground_dem.astype(np.float32)
    finite_mask = base_valid_mask & np.isfinite(raw_depth_m)
    too_deep_mask = finite_mask & (raw_depth_m > max_valid_depth_m)
    valid_depth_mask = finite_mask & (raw_depth_m >= min_valid_depth_m) & (raw_depth_m <= max_valid_depth_m)

    depth_map_m = np.full(surface_dem.shape, outside_value, dtype=np.float32)
    if negative_policy == "set_zero":
        shallow_or_negative = finite_mask & (raw_depth_m < min_valid_depth_m)
        depth_map_m[shallow_or_negative] = 0.0
    elif negative_policy == "invalid":
        pass
    else:
        raise ValueError(f"Unsupported negative_depth_policy: {negative_policy}")
    depth_map_m[valid_depth_mask] = raw_depth_m[valid_depth_mask].astype(np.float32)

    warnings: list[str] = []
    water_region_cell_count = int(np.count_nonzero(water_region_mask))
    valid_depth_cell_count = int(np.count_nonzero(valid_depth_mask))
    valid_ratio = (
        float(valid_depth_cell_count / water_region_cell_count)
        if water_region_cell_count > 0
        else 0.0
    )
    if water_region_cell_count == 0:
        warnings.append("low_data_quality: water_region_mask has zero cells")
    if valid_depth_cell_count == 0:
        warnings.append("low_data_quality: no valid surface-difference depth cells")
    if valid_ratio < 0.05:
        warnings.append("low_data_quality: valid_depth_ratio_in_water_region is below 0.05")
    too_deep_count = int(np.count_nonzero(too_deep_mask))
    if too_deep_count > 0:
        warnings.append(f"{too_deep_count} cells exceeded max_valid_depth_m and were marked invalid")

    stats = _stats_cm(depth_map_m, valid_depth_mask)
    output_depth_map = hydrology_dir / "surface_water_depth_map.npy"
    output_valid_mask = hydrology_dir / "surface_water_depth_valid_mask.npy"
    output_json = json_dir / f"surface_water_depth_result_{output_name}.json"
    output_figure = figure_dir / f"surface_water_depth_heatmap_{output_name}.png"
    np.save(output_depth_map, depth_map_m)
    np.save(output_valid_mask, valid_depth_mask)
    save_depth_heatmap(
        depth_map_m,
        valid_depth_mask,
        output_figure,
        f"S4-real Surface Difference Depth - {output_name}",
    )

    result = {
        "stage": "S4_real_surface_difference_depth",
        "case_name": output_name,
        "depth_source": "offline_lidar_surface_dem_difference",
        "source_surface_dem": str(surface_dem_path),
        "source_ground_dem": str(ground_dem_path),
        "source_water_region_mask": str(water_mask_path),
        "known_depth_cm": case_config.get("known_depth_cm"),
        "valid_depth_cell_count": valid_depth_cell_count,
        "water_region_cell_count": water_region_cell_count,
        "valid_depth_ratio_in_water_region": valid_ratio,
        "max_depth_cm": stats["max_depth_cm"],
        "mean_depth_cm": stats["mean_depth_cm"],
        "median_depth_cm": stats["median_depth_cm"],
        "min_depth_cm": stats["min_depth_cm"],
        "min_valid_depth_m": min_valid_depth_m,
        "max_valid_depth_m": max_valid_depth_m,
        "negative_depth_policy": negative_policy,
        "note": NOTE,
        "warning": warnings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_files": {
            "surface_water_depth_map": str(output_depth_map),
            "surface_water_depth_valid_mask": str(output_valid_mask),
            "surface_water_depth_result_json": str(output_json),
            "surface_water_depth_heatmap": str(output_figure),
        },
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[surface_depth] case_name: {output_name}")
    print(f"[surface_depth] valid_depth_cell_count: {valid_depth_cell_count}")
    print(f"[surface_depth] valid_depth_ratio_in_water_region: {valid_ratio:.4f}")
    print(f"[surface_depth] mean_depth_cm: {_fmt(stats['mean_depth_cm'])}")
    print(f"[surface_depth] median_depth_cm: {_fmt(stats['median_depth_cm'])}")
    print(f"[surface_depth] max_depth_cm: {_fmt(stats['max_depth_cm'])}")
    if warnings:
        print(f"[surface_depth][WARN] {'; '.join(warnings)}")
    print("[surface_depth] output paths:")
    for path in result["output_files"].values():
        print(f"  - {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Invert S4-real depth from surface DEM minus ground DEM.")
    parser.add_argument("--config", required=True, help="Path to configs/surface_dem_config.yaml")
    parser.add_argument("--case", help="Case name, e.g. water_sim_13cm_001")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    invert_surface_depth(args.config, args.project_root, args.case)


if __name__ == "__main__":
    main()
