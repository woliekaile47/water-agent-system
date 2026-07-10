#!/usr/bin/env python3
"""Estimate water level from ray/DEM shoreline intersections without GT."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.hydrology.estimate_water_level_from_boundary import robust_filter


def _stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "median_m": None, "mad_m": None, "iqr_m": None, "std_m": None}
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    return {
        "count": int(values.size),
        "min_m": float(np.min(values)),
        "max_m": float(np.max(values)),
        "median_m": median,
        "mad_m": float(np.median(np.abs(values - median))),
        "iqr_m": float(q75 - q25),
        "std_m": float(np.std(values)),
    }


def _estimate(values: np.ndarray, config: dict[str, Any]) -> float:
    method = str(config.get("method", "robust_median"))
    if method == "median":
        return float(np.median(values))
    if method == "trimmed_median":
        ordered = np.sort(values)
        trim = int(np.floor(ordered.size * float(config.get("trim_fraction", 0.10))))
        trimmed = ordered[trim : ordered.size - trim] if trim > 0 else ordered
        return float(np.median(trimmed if trimmed.size else ordered))
    if method == "robust_median":
        return float(np.median(values))
    raise ValueError(f"Unsupported shoreline water-level method: {method}")


def estimate_water_level_from_shoreline(
    intersections: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Estimate per image component, then combine by valid sample weight."""
    by_component: dict[int, list[float]] = {}
    for record in intersections:
        value = float(record["dem_height_m"])
        if np.isfinite(value):
            by_component.setdefault(int(record["component_index"]), []).append(value)
    minimum = int(config.get("min_samples_per_component", 10))
    results: list[dict[str, Any]] = []
    levels: list[float] = []
    weights: list[int] = []
    for component_index in sorted(by_component):
        raw = np.asarray(by_component[component_index], dtype=np.float64)
        filtered, filter_diagnostics = robust_filter(raw, config)
        if filtered.size < minimum:
            results.append({
                "component_index": component_index,
                "accepted": False,
                "reason": f"filtered_samples={filtered.size} < {minimum}",
                "raw_stats": _stats(raw),
                "filtered_stats": _stats(filtered),
                "outlier_filter": filter_diagnostics,
            })
            continue
        level = _estimate(filtered, config)
        levels.append(level)
        weights.append(int(filtered.size))
        results.append({
            "component_index": component_index,
            "accepted": True,
            "estimated_water_level_m": level,
            "raw_stats": _stats(raw),
            "filtered_stats": _stats(filtered),
            "outlier_filter": filter_diagnostics,
        })
    if not levels:
        raise ValueError("No image shoreline component has enough valid ray/DEM intersections")
    order = np.argsort(np.asarray(levels))
    sorted_levels = np.asarray(levels)[order]
    sorted_weights = np.asarray(weights)[order]
    selected = int(np.searchsorted(np.cumsum(sorted_weights), np.sum(sorted_weights) / 2.0, side="left"))
    level = float(sorted_levels[min(selected, sorted_levels.size - 1)])
    accepted_values = np.asarray(
        [record["dem_height_m"] for record in intersections if np.isfinite(float(record["dem_height_m"]))],
        dtype=np.float64,
    )
    filtered_all, _ = robust_filter(accepted_values, config)
    global_stats = _stats(filtered_all)
    diagnostics = {
        "method": str(config.get("method", "robust_median")),
        "estimated_water_level_m": level,
        "valid_shoreline_sample_count": int(sum(weights)),
        "shoreline_height_mad_m": global_stats["mad_m"],
        "shoreline_height_iqr_m": global_stats["iqr_m"],
        "shoreline_height_std_m": global_stats["std_m"],
        "accepted_component_count": len(levels),
        "component_results": results,
        "iteration_count": 1,
        "water_level_delta_m": 0.0,
        "water_level_converged": True,
        "convergence_note": "Direct ray/DEM shoreline solve; no GT-guided optimization or per-case iteration.",
    }
    return level, diagnostics
