#!/usr/bin/env python3
"""Phase 2A quality gate independent of simulation Ground Truth answers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def evaluate_quality_gate(
    camera_mask: np.ndarray,
    predicted_mask: np.ndarray,
    predicted_depth: np.ndarray,
    projection: dict[str, Any],
    boundary: dict[str, Any],
    water_result: dict[str, Any],
    config: dict[str, Any],
    required_files: list[str | Path] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    metrics = {
        "camera_mask_pixel_count": int(np.count_nonzero(camera_mask)),
        "projection_coverage": float(projection.get("projection_coverage", 0.0)),
        "predicted_water_cell_count": int(np.count_nonzero(predicted_mask)),
        "component_count": int(boundary.get("component_count", 0)),
        "largest_component_ratio": float(boundary.get("largest_component_ratio", 0.0)),
        "valid_boundary_sample_count": int(boundary.get("valid_boundary_sample_count", 0)),
        "boundary_height_mad_m": boundary.get("boundary_height_mad_m"),
        "boundary_height_iqr_m": boundary.get("boundary_height_iqr_m"),
        "all_components_bracket_valid": bool(boundary.get("all_components_bracket_valid", False)),
        "estimated_water_level_m": boundary.get("estimated_water_level_m"),
        "max_depth_m": water_result.get("max_depth_m"),
        "negative_depth_count": int(water_result.get("negative_depth_count", 0)),
        "water_area_m2": water_result.get("water_area_m2"),
        "water_volume_m3": water_result.get("water_volume_m3"),
        "inf_depth_count": int(water_result.get("inf_depth_count", 0)),
    }
    if metrics["camera_mask_pixel_count"] <= 0:
        reasons.append("camera_mask_empty")
    if metrics["projection_coverage"] < float(config["min_projection_coverage"]):
        reasons.append("projection_coverage_below_threshold")
    if metrics["predicted_water_cell_count"] <= 0:
        reasons.append("predicted_dem_mask_empty")
    if not (1 <= metrics["component_count"] <= int(config["max_component_count"])):
        reasons.append("component_count_out_of_range")
    if metrics["largest_component_ratio"] < float(config["min_largest_component_ratio"]):
        reasons.append("largest_component_ratio_below_threshold")
    if metrics["valid_boundary_sample_count"] < int(config["min_boundary_samples"]):
        reasons.append("boundary_sample_count_below_threshold")
    mad = metrics["boundary_height_mad_m"]
    iqr = metrics["boundary_height_iqr_m"]
    if mad is None or not np.isfinite(mad) or float(mad) > float(config["max_boundary_mad_m"]):
        reasons.append("boundary_height_mad_above_threshold")
    if iqr is None or not np.isfinite(iqr) or float(iqr) > float(config["max_boundary_iqr_m"]):
        reasons.append("boundary_height_iqr_above_threshold")
    if bool(config.get("require_valid_inner_outer_bracket", True)) and not metrics["all_components_bracket_valid"]:
        reasons.append("inner_outer_boundary_bracket_invalid")
    level = metrics["estimated_water_level_m"]
    if level is None or not np.isfinite(level):
        reasons.append("estimated_water_level_nonfinite")
    maximum = metrics["max_depth_m"]
    if maximum is None or not np.isfinite(maximum) or float(maximum) > float(config["max_physical_depth_m"]):
        reasons.append("maximum_depth_exceeds_physical_limit")
    if metrics["negative_depth_count"] != 0:
        reasons.append("negative_depth_detected")
    if metrics["water_area_m2"] is None or float(metrics["water_area_m2"]) < 0:
        reasons.append("negative_or_missing_area")
    if metrics["water_volume_m3"] is None or float(metrics["water_volume_m3"]) < 0:
        reasons.append("negative_or_missing_volume")
    if metrics["inf_depth_count"] != 0 or np.isinf(np.asarray(predicted_depth)).any():
        reasons.append("infinite_depth_detected")
    missing_files = [str(path) for path in (required_files or []) if not Path(path).is_file()]
    if missing_files:
        reasons.append("prediction_result_files_incomplete")
    metrics["missing_required_files"] = missing_files
    return {
        "status": "reject" if reasons else "pass",
        "reasons": reasons,
        "metrics": metrics,
        "eligible_for_downstream": False,
        "downstream_block_reason": "Phase 2A simulation evaluation is not connected to formal S5-S8",
    }
