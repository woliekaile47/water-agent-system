#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4: Invert water depth from mapped region and ground DEM."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.fusion.map_mask_to_dem import load_json, load_yaml, resolve_project_path
from src.hydrology.visualize_depth_map import save_depth_heatmap


def invert_water_depth(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)
    mapping_method = str(config.get("mapping_method", "region_level_manual"))
    water_surface_config = config["water_surface"]
    mode = str(water_surface_config.get("mode", "configured_depth"))
    if mode != "configured_depth":
        raise ValueError(f"S4 MVP 当前只支持 water_surface.mode=configured_depth，不支持: {mode}")

    configured_depth_cm = float(water_surface_config["configured_depth_cm"])
    ground_dem_path = resolve_project_path(root, config["dem"]["ground_dem_interpolated_path"])
    ground_valid_mask_path = resolve_project_path(root, config["dem"]["ground_dem_valid_mask_path"])
    dem_metadata_path = resolve_project_path(root, config["dem"]["metadata_path"])
    source_mask_path = root / "data" / "fusion" / "water_region_mask.npy"
    mapping_json_path = root / "data" / "fusion" / "mask_to_dem_mapping.json"
    if not source_mask_path.exists():
        raise FileNotFoundError(f"water_region_mask 不存在: {source_mask_path}。请先运行 --stage mask_to_dem。")

    ground_dem = np.load(ground_dem_path)
    ground_valid_mask = np.load(ground_valid_mask_path).astype(bool)
    water_region_mask = np.load(source_mask_path).astype(bool)
    dem_metadata = load_json(dem_metadata_path)
    mapping_metadata = load_json(mapping_json_path) if mapping_json_path.exists() else {}

    if ground_dem.shape != water_region_mask.shape:
        raise ValueError(f"ground_dem 和 water_region_mask shape 不一致: {ground_dem.shape} vs {water_region_mask.shape}")
    region_valid_mask = water_region_mask & np.isfinite(ground_dem)
    if not np.any(region_valid_mask):
        raise RuntimeError("water_region_mask 内没有有效 ground DEM 值，无法生成水深图")

    ground_values = ground_dem[region_valid_mask]
    ground_z_min = float(np.min(ground_values))
    ground_z_max = float(np.max(ground_values))
    ground_z_median = float(np.median(ground_values))
    water_surface_z_m = ground_z_median + configured_depth_cm / 100.0

    depth_map_m = np.zeros_like(ground_dem, dtype=np.float32)
    depth_region = np.maximum(water_surface_z_m - ground_dem[region_valid_mask], 0.0)
    depth_map_m[region_valid_mask] = depth_region.astype(np.float32)
    depth_valid_mask = region_valid_mask & (depth_map_m > 0)
    valid_depth_values_cm = depth_map_m[depth_valid_mask] * 100.0
    if valid_depth_values_cm.size == 0:
        max_depth_cm = mean_depth_cm = median_depth_cm = 0.0
    else:
        max_depth_cm = float(np.max(valid_depth_values_cm))
        mean_depth_cm = float(np.mean(valid_depth_values_cm))
        median_depth_cm = float(np.median(valid_depth_values_cm))

    hydrology_dir = root / "data" / "hydrology"
    figure_dir = root / "outputs" / "figures"
    json_dir = root / "outputs" / "json"
    hydrology_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    output_depth_map = hydrology_dir / "water_depth_map.npy"
    output_depth_valid_mask = hydrology_dir / "water_depth_valid_mask.npy"
    output_heatmap = figure_dir / "water_depth_heatmap.png"
    output_json = json_dir / "water_depth_result.json"

    np.save(output_depth_map, depth_map_m)
    np.save(output_depth_valid_mask, depth_valid_mask)
    save_depth_heatmap(depth_map_m, depth_valid_mask, output_heatmap, dem_metadata)

    result = {
        "stage": "S4_water_depth_inversion",
        "mapping_method": mapping_method,
        "depth_method": "configured_depth_region_level_mvp",
        "simulation_label": "configured_mvp_simulation",
        "configured_depth_cm": configured_depth_cm,
        "ground_z_min_in_region": ground_z_min,
        "ground_z_max_in_region": ground_z_max,
        "ground_z_median_in_region": ground_z_median,
        "water_surface_z_m": float(water_surface_z_m),
        "max_depth_cm": max_depth_cm,
        "mean_depth_cm": mean_depth_cm,
        "median_depth_cm": median_depth_cm,
        "valid_depth_cell_count": int(np.count_nonzero(depth_valid_mask)),
        "water_region_cell_count": int(np.count_nonzero(water_region_mask)),
        "source_ground_dem": str(ground_dem_path),
        "source_mask": str(source_mask_path),
        "source_mapping": str(mapping_json_path),
        "mapping_grid_index_range": mapping_metadata.get("dem_grid_index_range"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "configured_mvp_simulation: S4 MVP uses configured depth to validate "
            "mask-to-DEM and depth-map pipeline. It is not final real-water measurement."
        ),
        "output_files": {
            "water_depth_map": str(output_depth_map),
            "water_depth_valid_mask": str(output_depth_valid_mask),
            "water_depth_heatmap": str(output_heatmap),
            "water_depth_result": str(output_json),
        },
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[S4][water_depth] configured depth: {configured_depth_cm:.2f} cm")
    print(f"[S4][water_depth] max depth cm: {max_depth_cm:.2f}")
    print(f"[S4][water_depth] mean depth cm: {mean_depth_cm:.2f}")
    print(f"[S4][water_depth] median depth cm: {median_depth_cm:.2f}")
    print("[S4][water_depth] output files:")
    for path in result["output_files"].values():
        print(f"  - {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="S4 configured-depth water depth inversion.")
    parser.add_argument("--config", required=True, help="Path to configs/roi_mapping.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    invert_water_depth(args.config, args.project_root)


if __name__ == "__main__":
    main()
