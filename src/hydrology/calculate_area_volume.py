#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S5: Calculate water area and volume from the S4 depth map."""

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

from src.fusion.map_mask_to_dem import load_json, load_yaml, resolve_project_path


def _read_grid_size_m(dem_metadata: dict[str, Any]) -> tuple[float, str]:
    for key in ("grid_size", "resolution", "cell_size", "cell_size_m"):
        value = dem_metadata.get(key)
        if value is not None:
            return float(value), ""
    return 0.1, "DEM metadata has no grid_size/resolution/cell_size/cell_size_m; defaulted to 0.1 m."


def _depth_stats_cm(values_cm: np.ndarray) -> dict[str, float]:
    if values_cm.size == 0:
        return {
            "max_depth_cm": 0.0,
            "mean_depth_cm": 0.0,
            "median_depth_cm": 0.0,
            "min_depth_cm": 0.0,
        }
    return {
        "max_depth_cm": float(np.max(values_cm)),
        "mean_depth_cm": float(np.mean(values_cm)),
        "median_depth_cm": float(np.median(values_cm)),
        "min_depth_cm": float(np.min(values_cm)),
    }


def calculate_area_volume(
    config_path: str | Path,
    project_root: str | Path,
    min_valid_depth_cm: float = 0.5,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)

    depth_map_path = root / "data" / "hydrology" / "water_depth_map.npy"
    valid_mask_path = root / "data" / "hydrology" / "water_depth_valid_mask.npy"
    s4_result_path = root / "outputs" / "json" / "water_depth_result.json"
    dem_metadata_path = resolve_project_path(root, config["dem"]["metadata_path"])

    for path in (depth_map_path, valid_mask_path, s4_result_path, dem_metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Required S5 input does not exist: {path}")

    depth_map_m = np.load(depth_map_path)
    water_depth_valid_mask = np.load(valid_mask_path).astype(bool)
    if depth_map_m.shape != water_depth_valid_mask.shape:
        raise ValueError(
            "water_depth_map and water_depth_valid_mask shapes do not match: "
            f"{depth_map_m.shape} vs {water_depth_valid_mask.shape}"
        )

    s4_result = load_json(s4_result_path)
    dem_metadata = load_json(dem_metadata_path)
    grid_size_m, warning = _read_grid_size_m(dem_metadata)
    cell_area_m2 = grid_size_m * grid_size_m

    depth_cm = depth_map_m.astype(np.float64) * 100.0
    valid_depth_mask = water_depth_valid_mask & np.isfinite(depth_cm) & (depth_cm > float(min_valid_depth_cm))
    valid_depth_values_m = depth_map_m[valid_depth_mask].astype(np.float64)
    valid_depth_values_cm = valid_depth_values_m * 100.0

    valid_depth_cell_count = int(np.count_nonzero(valid_depth_mask))
    water_area_m2 = float(valid_depth_cell_count * cell_area_m2)
    water_volume_m3 = float(np.sum(valid_depth_values_m * cell_area_m2))
    water_volume_liter = float(water_volume_m3 * 1000.0)
    stats = _depth_stats_cm(valid_depth_values_cm)

    hydrology_dir = root / "data" / "hydrology"
    json_dir = root / "outputs" / "json"
    hydrology_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    data_output_json = hydrology_dir / "water_area_volume_result.json"
    output_json = json_dir / "water_area_volume_result.json"

    result = {
        "stage": "S5_area_volume_calculation",
        "source_depth_map": str(depth_map_path),
        "source_valid_mask": str(valid_mask_path),
        "source_s4_result_json": str(s4_result_path),
        "source_dem_metadata": str(dem_metadata_path),
        "depth_method_from_s4": s4_result.get("depth_method"),
        "simulation_label_from_s4": s4_result.get("simulation_label", "configured_mvp_simulation"),
        "configured_depth_cm_from_s4": s4_result.get("configured_depth_cm"),
        "mvp_note": s4_result.get("note"),
        "min_valid_depth_cm": float(min_valid_depth_cm),
        "grid_size_m": grid_size_m,
        "cell_area_m2": cell_area_m2,
        "valid_depth_cell_count": valid_depth_cell_count,
        "water_area_m2": water_area_m2,
        "water_volume_m3": water_volume_m3,
        "water_volume_liter": water_volume_liter,
        "max_depth_cm": stats["max_depth_cm"],
        "mean_depth_cm": stats["mean_depth_cm"],
        "median_depth_cm": stats["median_depth_cm"],
        "min_depth_cm": stats["min_depth_cm"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "warning": warning,
        "output_files": {
            "data_result_json": str(data_output_json),
            "output_result_json": str(output_json),
        },
    }

    for path in (data_output_json, output_json):
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(f"[S5][area_volume] valid depth cell count: {valid_depth_cell_count}")
    print(f"[S5][area_volume] water area m2: {water_area_m2:.4f}")
    print(f"[S5][area_volume] water volume m3: {water_volume_m3:.4f}")
    print(f"[S5][area_volume] water volume liter: {water_volume_liter:.2f}")
    print(f"[S5][area_volume] max depth cm: {stats['max_depth_cm']:.2f}")
    print(f"[S5][area_volume] mean depth cm: {stats['mean_depth_cm']:.2f}")
    if warning:
        print(f"[S5][area_volume][warning] {warning}")
    print("[S5][area_volume] output files:")
    for path in result["output_files"].values():
        print(f"  - {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="S5 area and volume calculation from S4 water depth map.")
    parser.add_argument("--config", required=True, help="Path to configs/roi_mapping.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    parser.add_argument("--min-valid-depth-cm", type=float, default=0.5, help="Minimum valid depth threshold in cm")
    args = parser.parse_args()
    calculate_area_volume(args.config, args.project_root, args.min_valid_depth_cm)


if __name__ == "__main__":
    main()
