#!/usr/bin/env python3
"""Ground-Truth-independent quality gate for Phase 2B."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def evaluate_water_surface_aware_quality_gate(
    ray_diagnostics: dict[str, Any],
    shoreline_diagnostics: dict[str, Any],
    reconstruction_diagnostics: dict[str, Any],
    consistency: dict[str, Any],
    water_result: dict[str, Any],
    predicted_depth: np.ndarray,
    config: dict[str, Any],
    required_files: list[str | Path] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    metrics = {
        "shoreline_intersection_success_rate": float(ray_diagnostics.get("shoreline_intersection_success_rate", 0.0)),
        "camera_mask_edge_touch_ratio": float(ray_diagnostics.get("camera_mask_edge_touch_ratio", 1.0)),
        "camera_reprojection_iou": float(consistency.get("camera_reprojection_iou", 0.0)),
        "boundary_reprojection_p95_px": consistency.get("boundary_reprojection_p95_px"),
        "water_surface_projection_coverage": float(consistency.get("water_surface_projection_coverage", 0.0)),
        "candidate_basin_count": int(reconstruction_diagnostics.get("candidate_basin_count", 0)),
        "selected_basin_count": int(reconstruction_diagnostics.get("selected_basin_count", 0)),
        "ambiguous_candidate_basins": bool(reconstruction_diagnostics.get("ambiguous_candidate_basins", True)),
        "ambiguous_candidate_basin_count": int(reconstruction_diagnostics.get("ambiguous_candidate_basin_count", 0)),
        "unobserved_candidate_basin_count": int(reconstruction_diagnostics.get("unobserved_candidate_basin_count", 0)),
        "camera_observable_candidate_basin_count": int(
            reconstruction_diagnostics.get("camera_observable_candidate_basin_count", 0)
        ),
        "seed_valid": bool(reconstruction_diagnostics.get("seed_valid", False)),
        "water_level_converged": bool(shoreline_diagnostics.get("water_level_converged", False)),
        "estimated_water_level_m": shoreline_diagnostics.get("estimated_water_level_m"),
        "valid_shoreline_sample_count": int(shoreline_diagnostics.get("valid_shoreline_sample_count", 0)),
        "shoreline_height_mad_m": shoreline_diagnostics.get("shoreline_height_mad_m"),
        "shoreline_height_iqr_m": shoreline_diagnostics.get("shoreline_height_iqr_m"),
        "max_depth_m": water_result.get("max_depth_m"),
        "negative_depth_count": int(water_result.get("negative_depth_count", 0)),
        "inf_depth_count": int(water_result.get("inf_depth_count", 0)),
    }
    if metrics["shoreline_intersection_success_rate"] < float(config["min_shoreline_intersection_rate"]):
        reasons.append("shoreline_intersection_rate_below_threshold")
    if metrics["camera_mask_edge_touch_ratio"] > float(config["max_camera_mask_edge_touch_ratio"]):
        reasons.append("camera_mask_touches_image_edge_excessively")
    if metrics["camera_reprojection_iou"] < float(config["min_camera_reprojection_iou"]):
        reasons.append("camera_reprojection_iou_below_threshold")
    boundary_error = metrics["boundary_reprojection_p95_px"]
    if boundary_error is None or not np.isfinite(boundary_error) or float(boundary_error) > float(config["max_boundary_reprojection_p95_px"]):
        reasons.append("boundary_reprojection_error_above_threshold")
    if metrics["candidate_basin_count"] > int(config["max_candidate_basin_count"]):
        reasons.append("too_many_candidate_basins")
    if metrics["ambiguous_candidate_basins"]:
        reasons.append("ambiguous_candidate_basin")
    if metrics["selected_basin_count"] < 1:
        reasons.append("no_selected_basin")
    if metrics["unobserved_candidate_basin_count"] > 0:
        reasons.append("candidate_basin_outside_camera_coverage")
    if not metrics["seed_valid"]:
        reasons.append("invalid_reconstruction_seed")
    if not metrics["water_level_converged"]:
        reasons.append("water_level_estimation_not_converged")
    if metrics["water_surface_projection_coverage"] < float(config["min_water_surface_projection_coverage"]):
        reasons.append("predicted_water_region_outside_camera_coverage")
    level = metrics["estimated_water_level_m"]
    if level is None or not np.isfinite(level):
        reasons.append("estimated_water_level_nonfinite")
    if metrics["valid_shoreline_sample_count"] < int(config["min_valid_shoreline_samples"]):
        reasons.append("shoreline_sample_count_below_threshold")
    mad = metrics["shoreline_height_mad_m"]
    iqr = metrics["shoreline_height_iqr_m"]
    if mad is None or not np.isfinite(mad) or float(mad) > float(config["max_shoreline_mad_m"]):
        reasons.append("shoreline_height_mad_above_threshold")
    if iqr is None or not np.isfinite(iqr) or float(iqr) > float(config["max_shoreline_iqr_m"]):
        reasons.append("shoreline_height_iqr_above_threshold")
    maximum = metrics["max_depth_m"]
    if maximum is None or not np.isfinite(maximum) or float(maximum) > float(config["max_physical_depth_m"]):
        reasons.append("maximum_depth_exceeds_physical_limit")
    depth = np.asarray(predicted_depth)
    if metrics["negative_depth_count"] or np.any(depth[np.isfinite(depth)] < 0.0):
        reasons.append("negative_depth_detected")
    if metrics["inf_depth_count"] or np.isinf(depth).any():
        reasons.append("infinite_depth_detected")
    missing = [str(path) for path in (required_files or []) if not Path(path).is_file()]
    if missing:
        reasons.append("prediction_result_files_incomplete")
    metrics["missing_required_files"] = missing
    global_scope_reasons = {
        "ambiguous_candidate_basin",
        "candidate_basin_outside_camera_coverage",
    }
    observable_region_reasons = [reason for reason in reasons if reason not in global_scope_reasons]
    observable_region_result_valid = bool(
        metrics["selected_basin_count"] > 0 and not observable_region_reasons
    )
    has_unobservable_scope = bool(
        metrics["unobserved_candidate_basin_count"] > 0
        or metrics["ambiguous_candidate_basin_count"] > 0
    )
    if not observable_region_result_valid:
        global_estimate_status = "unavailable"
    elif has_unobservable_scope:
        global_estimate_status = "partial"
    else:
        global_estimate_status = "complete"
    result_semantics = "global_estimate" if global_estimate_status == "complete" else "observable_region_estimate"
    area_volume_semantics = "complete_estimate" if global_estimate_status == "complete" else "observable_lower_bound"
    return {
        "status": "reject" if reasons else "pass",
        "reasons": reasons,
        "metrics": metrics,
        "observation_scope": "camera_observable_region",
        "global_estimate_status": global_estimate_status,
        "observable_region_result_valid": observable_region_result_valid,
        "unobservable_candidate_basin_count": metrics["unobserved_candidate_basin_count"],
        "ambiguous_candidate_basin_count": metrics["ambiguous_candidate_basin_count"],
        "camera_observable_candidate_basin_count": metrics["camera_observable_candidate_basin_count"],
        "result_semantics": result_semantics,
        "area_volume_semantics": area_volume_semantics,
        "eligible_for_downstream": False,
        "downstream_block_reason": "Phase 2B simulation experiment is not connected to formal S5-S8",
    }
