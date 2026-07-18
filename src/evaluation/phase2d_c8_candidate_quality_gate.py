#!/usr/bin/env python3
"""Offline Phase 2D-C-8 candidate gate using prediction-side metrics only."""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def evaluate_candidate_gate(
    frame_metrics: dict[str, Any],
    sequence_metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one frozen frame without accepting Ground Truth inputs."""
    frame_cfg = config["frame_thresholds"]
    sequence_cfg = config["sequence_thresholds"]
    hard_reasons: list[str] = []
    visible_reasons: list[str] = []
    temporal_reasons: list[str] = []
    global_scope_reasons: list[str] = []
    warnings: list[str] = []

    if not bool(frame_metrics.get("geometry_available", False)):
        hard_reasons.append("geometry_unavailable")
    if int(frame_metrics.get("selected_basin_count", 0)) < 1:
        hard_reasons.append("no_selected_basin")
    if not _finite(frame_metrics.get("estimated_water_level_m")):
        hard_reasons.append("estimated_water_level_nonfinite")
    maximum_depth_cm = frame_metrics.get("max_depth_cm")
    if not _finite(maximum_depth_cm):
        hard_reasons.append("maximum_depth_nonfinite")
    elif float(maximum_depth_cm) < 0.0:
        hard_reasons.append("negative_depth_detected")
    elif float(maximum_depth_cm) > float(frame_cfg["max_physical_depth_m"]) * 100.0:
        hard_reasons.append("maximum_depth_exceeds_physical_limit")

    intersection_rate = float(frame_metrics.get("shoreline_intersection_success_rate", 0.0))
    if intersection_rate < float(frame_cfg["min_shoreline_intersection_success_rate"]):
        visible_reasons.append("shoreline_intersection_rate_below_candidate_threshold")
    seed_rate = float(frame_metrics.get("seed_ray_success_ratio", 0.0))
    if seed_rate < float(frame_cfg["min_seed_ray_success_ratio"]):
        visible_reasons.append("seed_ray_success_rate_below_candidate_threshold")
    sample_count = int(frame_metrics.get("filtered_shoreline_sample_count", 0))
    if sample_count < int(frame_cfg["min_valid_shoreline_samples"]):
        visible_reasons.append("shoreline_sample_count_below_candidate_threshold")
    mad = frame_metrics.get("filtered_shoreline_mad_m")
    if not _finite(mad) or float(mad) > float(frame_cfg["max_shoreline_mad_m"]):
        visible_reasons.append("shoreline_height_mad_above_candidate_threshold")
    iqr = frame_metrics.get("filtered_shoreline_iqr_m")
    if not _finite(iqr) or float(iqr) > float(frame_cfg["max_shoreline_iqr_m"]):
        visible_reasons.append("shoreline_height_iqr_above_candidate_threshold")
    camera_iou = frame_metrics.get("camera_reprojection_iou")
    if not _finite(camera_iou) or float(camera_iou) < float(frame_cfg["min_camera_reprojection_iou"]):
        visible_reasons.append("camera_reprojection_iou_below_candidate_threshold")

    boundary_p95 = frame_metrics.get("outer_boundary_reprojection_p95_px")
    if not _finite(boundary_p95):
        warnings.append("outer_boundary_reprojection_p95_unavailable")
    elif float(boundary_p95) > float(frame_cfg["advisory_outer_boundary_p95_px"]):
        warnings.append("outer_boundary_reprojection_p95_above_advisory_threshold")

    level_std_cm = float(sequence_metrics["estimated_water_level_m"]["std"]) * 100.0
    adjacent_p95_cm = float(sequence_metrics["adjacent_absolute_water_level_change_m"]["p95"]) * 100.0
    area_cv = float(sequence_metrics["water_area_m2"]["coefficient_of_variation"])
    volume_cv = float(sequence_metrics["water_volume_m3"]["coefficient_of_variation"])
    if level_std_cm > float(sequence_cfg["max_water_level_window_std_cm"]):
        temporal_reasons.append("water_level_window_std_above_candidate_threshold")
    if adjacent_p95_cm > float(sequence_cfg["max_adjacent_water_level_change_p95_cm"]):
        temporal_reasons.append("adjacent_water_level_change_p95_above_candidate_threshold")
    if area_cv > float(sequence_cfg["max_water_area_coefficient_of_variation"]):
        temporal_reasons.append("water_area_cv_above_candidate_threshold")
    if volume_cv > float(sequence_cfg["max_water_volume_coefficient_of_variation"]):
        temporal_reasons.append("water_volume_cv_above_candidate_threshold")

    if int(frame_metrics.get("unobserved_candidate_basin_count", 0)) > 0:
        global_scope_reasons.append("candidate_basin_outside_camera_coverage")
    if int(frame_metrics.get("ambiguous_candidate_basin_count", 0)) > 0:
        global_scope_reasons.append("ambiguous_candidate_basin")

    visible_reject_reasons = hard_reasons + visible_reasons + temporal_reasons
    camera_visible_status = "reject" if visible_reject_reasons else "pass"
    if camera_visible_status == "reject":
        global_scene_status = "unavailable"
        result_semantics = "unavailable"
    elif global_scope_reasons:
        global_scene_status = "partial"
        result_semantics = "camera_visible_estimate"
    else:
        global_scene_status = "complete"
        result_semantics = "global_scene_estimate"

    return {
        "protocol_version": config["protocol_version"],
        "status": "offline_research_candidate",
        "frame_index": int(frame_metrics["frame_index"]),
        "camera_visible_status": camera_visible_status,
        "global_scene_status": global_scene_status,
        "result_semantics": result_semantics,
        "hard_safety_reasons": hard_reasons,
        "camera_geometry_reasons": visible_reasons,
        "temporal_reasons": temporal_reasons,
        "global_scope_reasons": global_scope_reasons,
        "warnings": warnings,
        "visible_reject_reasons": visible_reject_reasons,
        "boundary_metric_rejected_by_itself": False,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
    }
