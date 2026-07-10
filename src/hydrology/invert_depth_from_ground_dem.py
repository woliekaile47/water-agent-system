#!/usr/bin/env python3
"""Invert water depth, area, and volume from an estimated level and dry DEM."""

from __future__ import annotations

from typing import Any

import numpy as np


def invert_depth_from_ground_dem(
    ground_dem: np.ndarray,
    predicted_dem_mask: np.ndarray,
    predicted_water_level_m: float,
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dem = np.asarray(ground_dem, dtype=np.float64)
    mask = np.asarray(predicted_dem_mask, dtype=bool)
    if dem.shape != mask.shape or dem.ndim != 2:
        raise ValueError("ground_dem and predicted_dem_mask must be matching 2D arrays")
    if not np.isfinite(predicted_water_level_m):
        raise ValueError("predicted_water_level_m must be finite")
    if not np.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("cell_size_m must be finite and positive")

    valid_dem = np.isfinite(dem)
    valid_water = mask & valid_dem
    depth = np.zeros(dem.shape, dtype=np.float32)
    depth[~valid_dem] = np.nan
    depth[valid_water] = np.maximum(0.0, predicted_water_level_m - dem[valid_water]).astype(np.float32)
    if np.any(depth[valid_dem] < 0.0) or np.isinf(depth).any():
        raise RuntimeError("Depth inversion produced invalid values")

    values = depth[valid_water].astype(np.float64)
    cell_area = float(cell_size_m * cell_size_m)
    area = float(np.count_nonzero(valid_water) * cell_area)
    volume = float(np.sum(values) * cell_area)
    result = {
        "predicted_water_level_m": float(predicted_water_level_m),
        "cell_size_m": float(cell_size_m),
        "cell_area_m2": cell_area,
        "valid_dem_cell_count": int(np.count_nonzero(valid_dem)),
        "predicted_water_cell_count": int(np.count_nonzero(valid_water)),
        "invalid_dem_in_mask_count": int(np.count_nonzero(mask & ~valid_dem)),
        "water_area_m2": area,
        "water_volume_m3": volume,
        "water_volume_l": float(volume * 1000.0),
        "max_depth_m": float(np.max(values)) if values.size else 0.0,
        "mean_depth_m": float(np.mean(values)) if values.size else 0.0,
        "median_depth_m": float(np.median(values)) if values.size else 0.0,
        "max_depth_cm": float(np.max(values) * 100.0) if values.size else 0.0,
        "mean_depth_cm": float(np.mean(values) * 100.0) if values.size else 0.0,
        "median_depth_cm": float(np.median(values) * 100.0) if values.size else 0.0,
        "negative_depth_count": int(np.count_nonzero(depth[valid_dem] < 0.0)),
        "inf_depth_count": int(np.count_nonzero(np.isinf(depth))),
    }
    return depth, valid_water, result
