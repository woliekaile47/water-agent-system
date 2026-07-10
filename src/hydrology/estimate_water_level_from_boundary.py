#!/usr/bin/env python3
"""Robust water-level estimation from a predicted DEM-space water mask."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from src.hydrology.invert_boundary_waterline_depth import extract_boundary_mask


def connected_components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be 2D")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    visited = np.zeros(binary.shape, dtype=bool)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    components: list[np.ndarray] = []
    height, width = binary.shape
    for start_row, start_col in zip(*np.where(binary & ~visited)):
        if visited[start_row, start_col]:
            continue
        component = np.zeros(binary.shape, dtype=bool)
        queue: deque[tuple[int, int]] = deque([(int(start_row), int(start_col))])
        visited[start_row, start_col] = True
        while queue:
            row, col = queue.popleft()
            component[row, col] = True
            for drow, dcol in offsets:
                next_row, next_col = row + drow, col + dcol
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and binary[next_row, next_col]
                    and not visited[next_row, next_col]
                ):
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))
        components.append(component)
    return components


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(result)
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                expanded |= padded[1 + drow : 1 + drow + result.shape[0], 1 + dcol : 1 + dcol + result.shape[1]]
        result = expanded
    return result


def robust_filter(values: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    method = str(config.get("outlier_method", "mad"))
    if finite.size == 0:
        return finite, {"method": method, "input_count": 0, "output_count": 0}
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    q25, q75 = np.percentile(finite, [25.0, 75.0])
    iqr = float(q75 - q25)
    if method == "mad":
        scale = 1.4826 * mad
        if scale <= 1e-12:
            keep = np.ones(finite.shape, dtype=bool)
        else:
            keep = np.abs(finite - median) <= float(config.get("mad_threshold", 3.5)) * scale
    elif method == "iqr":
        multiplier = float(config.get("iqr_multiplier", 1.5))
        keep = (finite >= q25 - multiplier * iqr) & (finite <= q75 + multiplier * iqr)
    elif method == "none":
        keep = np.ones(finite.shape, dtype=bool)
    else:
        raise ValueError(f"Unsupported outlier_method: {method}")
    filtered = finite[keep]
    return filtered, {
        "method": method,
        "input_count": int(finite.size),
        "output_count": int(filtered.size),
        "median_m": median,
        "mad_m": mad,
        "iqr_m": iqr,
        "std_m": float(np.std(finite)),
    }


def _sample_stats(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "min_m": None, "median_m": None, "max_m": None, "mad_m": None, "iqr_m": None, "std_m": None}
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    return {
        "count": int(values.size),
        "min_m": float(np.min(values)),
        "median_m": median,
        "max_m": float(np.max(values)),
        "mad_m": float(np.median(np.abs(values - median))),
        "iqr_m": float(q75 - q25),
        "std_m": float(np.std(values)),
    }


def _estimate_component_level(
    component: np.ndarray,
    ground_dem: np.ndarray,
    valid_dem: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    inner = extract_boundary_mask(component)
    outer = dilate_mask(component, int(config.get("outer_ring_cells", 1))) & ~component
    inner_values, inner_filter = robust_filter(ground_dem[inner & valid_dem], config)
    outer_values, outer_filter = robust_filter(ground_dem[outer & valid_dem], config)
    minimum = int(config.get("min_boundary_samples", 10))
    if inner_values.size < minimum:
        raise ValueError(f"valid inner boundary samples {inner_values.size} < {minimum}")
    method = str(config["method"])
    fallback_used = False
    bracket_valid = None
    bracket_inner_m = None
    bracket_outer_m = None
    if method == "inner_boundary_median":
        level = float(np.median(inner_values))
    elif method == "inner_boundary_high_quantile":
        level = float(np.quantile(inner_values, float(config.get("inner_high_quantile", 0.90))))
    elif method == "inner_outer_bracket_midpoint":
        bracket_inner_m = float(np.quantile(inner_values, float(config.get("inner_high_quantile", 0.90))))
        if outer_values.size >= minimum:
            bracket_outer_m = float(np.quantile(outer_values, float(config.get("outer_low_quantile", 0.10))))
            gap = bracket_outer_m - bracket_inner_m
            bracket_valid = bool(
                gap >= -float(config.get("bracket_order_tolerance_m", 0.01))
                and abs(gap) <= float(config.get("max_bracket_width_m", 0.10))
            )
        else:
            bracket_valid = False
        if bracket_valid:
            level = float((bracket_inner_m + bracket_outer_m) / 2.0)
        else:
            fallback_used = True
            level = bracket_inner_m
    else:
        raise ValueError(f"Unsupported water-level method: {method}")
    return level, {
        "method": method,
        "water_level_m": level,
        "inner_boundary_stats": _sample_stats(inner_values),
        "outer_boundary_stats": _sample_stats(outer_values),
        "inner_filter": inner_filter,
        "outer_filter": outer_filter,
        "bracket_inner_m": bracket_inner_m,
        "bracket_outer_m": bracket_outer_m,
        "bracket_valid": bracket_valid,
        "fallback_used": fallback_used,
    }


def estimate_water_level_from_boundary(
    predicted_mask: np.ndarray,
    ground_dem: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, np.ndarray, dict[str, Any]]:
    mask = np.asarray(predicted_mask, dtype=bool)
    dem = np.asarray(ground_dem, dtype=np.float64)
    if mask.shape != dem.shape:
        raise ValueError("predicted_mask and ground_dem shapes differ")
    valid_dem = np.isfinite(dem)
    components_before = connected_components(mask, int(config.get("connectivity", 8)))
    minimum_area = int(config.get("min_component_cells", 5))
    retained = [component for component in components_before if int(np.count_nonzero(component)) >= minimum_area]
    if not retained:
        raise ValueError("No predicted water component remains after area filtering")
    cleaned = np.logical_or.reduce(retained)
    component_results: list[dict[str, Any]] = []
    levels: list[float] = []
    weights: list[int] = []
    for index, component in enumerate(retained):
        level, details = _estimate_component_level(component, dem, valid_dem, config)
        cell_count = int(np.count_nonzero(component))
        details.update({"component_index": index, "cell_count": cell_count})
        component_results.append(details)
        levels.append(level)
        weights.append(cell_count)
    order = np.argsort(np.asarray(levels))
    sorted_levels = np.asarray(levels)[order]
    sorted_weights = np.asarray(weights)[order]
    selected = int(np.searchsorted(np.cumsum(sorted_weights), np.sum(sorted_weights) / 2.0, side="left"))
    water_level = float(sorted_levels[min(selected, sorted_levels.size - 1)])

    inner_all = extract_boundary_mask(cleaned) & valid_dem
    inner_values, _ = robust_filter(dem[inner_all], config)
    stats = _sample_stats(inner_values)
    largest_ratio = float(max(weights) / max(1, sum(weights)))
    diagnostics = {
        "method": str(config["method"]),
        "estimated_water_level_m": water_level,
        "component_count_before_filter": len(components_before),
        "component_count": len(retained),
        "removed_small_component_count": len(components_before) - len(retained),
        "largest_component_ratio": largest_ratio,
        "predicted_water_cell_count": int(np.count_nonzero(cleaned)),
        "valid_boundary_sample_count": int(stats["count"]),
        "boundary_height_mad_m": stats["mad_m"],
        "boundary_height_iqr_m": stats["iqr_m"],
        "boundary_height_std_m": stats["std_m"],
        "all_components_bracket_valid": bool(all(item["bracket_valid"] is not False for item in component_results)),
        "component_results": component_results,
    }
    return water_level, cleaned, diagnostics
