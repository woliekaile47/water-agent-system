#!/usr/bin/env python3
"""Prediction-side geometry for frozen SAM 2 video masks.

The module reuses the established C3C ray, water-level, reconstruction and
quality-gate implementations.  It never reads Ground Truth.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import cv2
import numpy as np

from src.evaluation.water_surface_aware_quality_gate import evaluate_water_surface_aware_quality_gate
from src.fusion.sam2_shoreline_geometry_adapter import (
    deterministic_seed_pixels,
    estimate_level_from_ray_records,
    extended_reprojection_consistency,
    intersect_pixel_rays,
    ray_hits_to_dem_seed_mask,
    reconstruct_seed_connected_lowland,
)
from src.fusion.water_surface_aware_mask_to_dem import reproject_water_surface
from src.hydrology.invert_boundary_waterline_depth import extract_boundary_mask
from src.hydrology.invert_depth_from_ground_dem import invert_depth_from_ground_dem


def video_prediction_semantics() -> dict[str, Any]:
    return {
        "semantic_label": "automatic_prompt_sam2_video_visible_water_candidate",
        "authoritative": False,
        "ground_truth_used": False,
        "result_scope": "single_video_frame_camera_visible_region",
        "area_volume_semantics": "camera_visible_candidate_estimate",
        "eligible_for_formal_s5_s8": False,
        "eligible_for_downstream": False,
        "result_note": (
            "Frozen automatic-prompt SAM2 video candidate plus dry Ground DEM; "
            "prediction-side temporal diagnostic only."
        ),
    }


def extract_main_outer_shoreline(
    mask: np.ndarray,
    sample_count: int = 128,
    connectivity: int = 8,
) -> dict[str, Any]:
    """Select the largest component and uniformly sample its external contour."""
    source = np.asarray(mask, dtype=bool)
    before = source.copy()
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        source.astype(np.uint8), connectivity=connectivity
    )
    component_count = int(labels_count - 1)
    if component_count < 1:
        raise ValueError("frozen SAM2 video mask is empty")
    areas = stats[1:, cv2.CC_STAT_AREA]
    selected_label = int(np.argmax(areas) + 1)
    selected = labels == selected_label
    contours, _ = cv2.findContours(
        selected.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("largest component has no external contour")
    contour = max(contours, key=cv2.contourArea)
    full = contour[:, 0, :].astype(np.float64)
    if full.shape[0] < 3:
        raise ValueError("external contour has fewer than three points")
    consecutive = np.linalg.norm(np.diff(full, axis=0), axis=1)
    keep = np.concatenate(([True], consecutive > 0.0))
    full = full[keep]
    if full.shape[0] < 3:
        raise ValueError("external contour is degenerate")
    closed = np.vstack((full, full[0]))
    segment = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(np.sum(segment))
    if perimeter <= 0.0:
        raise ValueError("external contour perimeter is zero")
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    targets = np.linspace(0.0, perimeter, int(sample_count), endpoint=False)
    sampled = np.empty((int(sample_count), 2), dtype=np.float64)
    for index, target in enumerate(targets):
        segment_index = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(segment) - 1)
        fraction = (target - cumulative[segment_index]) / max(segment[segment_index], 1e-12)
        sampled[index] = closed[segment_index] + fraction * (
            closed[segment_index + 1] - closed[segment_index]
        )
    if not np.array_equal(source, before):
        raise RuntimeError("shoreline extraction modified the frozen input mask")
    return {
        "selected_mask": selected,
        "full_shoreline_xy": full,
        "sampled_shoreline_xy": sampled,
        "component_count": component_count,
        "selected_component_pixels": int(np.count_nonzero(selected)),
        "excluded_fragment_pixels": int(np.count_nonzero(source & ~selected)),
        "full_shoreline_point_count": int(full.shape[0]),
        "sampled_shoreline_point_count": int(sampled.shape[0]),
        "perimeter_px": perimeter,
        "mean_sample_spacing_px": float(perimeter / int(sample_count)),
        "extraction_method": "largest_component_RETR_EXTERNAL_CHAIN_APPROX_NONE_arc_length_128",
        "morphology_applied": False,
        "hole_fill_applied": False,
        "ground_truth_used": False,
    }


def _edge_touch_ratio(mask: np.ndarray, edge_band: int) -> float:
    boundary = extract_boundary_mask(np.asarray(mask, dtype=bool))
    rows, columns = np.where(boundary)
    if rows.size == 0:
        return 0.0
    touches = (
        (rows < edge_band)
        | (rows >= mask.shape[0] - edge_band)
        | (columns < edge_band)
        | (columns >= mask.shape[1] - edge_band)
    )
    return float(np.count_nonzero(touches) / rows.size)


def _failure(reason: str, topology: dict[str, Any] | None = None) -> dict[str, Any]:
    semantics = video_prediction_semantics()
    return {
        **semantics,
        "available": False,
        "failure_reason": reason,
        "topology": topology,
        "quality_status": "reject",
        "geometry_diagnostic_readiness": "reject",
        "prediction_side_quality_gate": {
            "status": "reject",
            "reasons": [reason],
            "eligible_for_downstream": False,
        },
    }


def run_video_frame_geometry(
    frozen_mask: np.ndarray,
    frozen_positive_points_xy: np.ndarray,
    ground_dem: np.ndarray,
    sensors: dict[str, Any],
    mapping: dict[str, Any],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    """Run one frame through unchanged C3C geometry and existing gate thresholds."""
    semantics = video_prediction_semantics()
    try:
        topology = extract_main_outer_shoreline(frozen_mask, 128, int(mapping.get("connectivity", 8)))
    except ValueError as error:
        return _failure(str(error))
    selected = topology["selected_mask"]
    rays, ray_summary = intersect_pixel_rays(
        topology["sampled_shoreline_xy"], ground_dem, sensors, mapping, "video_shoreline"
    )
    ray_diagnostics = {
        **semantics,
        **ray_summary,
        "sampled_shoreline_ray_count": ray_summary["total_ray_count"],
        "shoreline_intersection_success_rate": ray_summary["intersection_success_ratio"],
        "camera_mask_edge_touch_ratio": _edge_touch_ratio(
            selected, max(1, int(mapping.get("image_edge_band_px", 2)))
        ),
    }
    if ray_summary["successful_intersection_count"] == 0:
        failed = _failure("no_valid_shoreline_intersections", topology)
        failed["ray_diagnostics"] = ray_diagnostics
        return failed
    try:
        level, water_level = estimate_level_from_ray_records(rays, mapping["shoreline_water_level"])
    except ValueError as error:
        failed = _failure(f"water_level_estimation_failed:{error}", topology)
        failed["ray_diagnostics"] = ray_diagnostics
        return failed
    water_level.update(semantics)
    seed_pixels, seed_sources = deterministic_seed_pixels(
        selected, np.asarray(frozen_positive_points_xy, dtype=np.float64)
    )
    seed_rays, seed_ray_summary = intersect_pixel_rays(
        seed_pixels, ground_dem, sensors, mapping, "video_seed"
    )
    for record, source in zip(seed_rays, seed_sources):
        record.update(source)
    seed_mask, mapped_seeds = ray_hits_to_dem_seed_mask(seed_rays, ground_dem.shape, sensors)
    predicted_mask, reconstruction = reconstruct_seed_connected_lowland(
        ground_dem, level, seed_mask, mapping["reconstruction"], selected, sensors
    )
    reconstruction.update({
        "camera_seed_pixel_count": int(seed_pixels.shape[0]),
        "successful_seed_ray_count": seed_ray_summary["successful_intersection_count"],
        "seed_ray_success_ratio": seed_ray_summary["intersection_success_ratio"],
        "seed_ray_failure_reasons": seed_ray_summary["intersection_failure_reasons"],
        "seed_records": mapped_seeds,
    })
    if not reconstruction.get("seed_valid") or not np.any(predicted_mask):
        failed = _failure("invalid_or_empty_reconstruction", topology)
        failed.update({
            "ray_diagnostics": ray_diagnostics,
            "water_level_estimation": water_level,
            "seed_diagnostics": reconstruction,
        })
        return failed
    cell_size = float(sensors["road"]["dem_resolution_m"])
    depth, predicted_mask, water_result = invert_depth_from_ground_dem(
        ground_dem, predicted_mask, level, cell_size
    )
    reprojected, projection = reproject_water_surface(predicted_mask, level, sensors)
    consistency = extended_reprojection_consistency(selected, reprojected, projection)
    consistency.update({
        "candidate_basin_count": reconstruction["candidate_basin_count"],
        "seed_validity": reconstruction["seed_valid"],
    })
    gate = evaluate_water_surface_aware_quality_gate(
        ray_diagnostics,
        water_level,
        reconstruction,
        consistency,
        water_result,
        depth,
        gate_config,
        required_files=None,
    )
    readiness = "ready" if gate["status"] == "pass" else "diagnostic_only"
    return {
        **semantics,
        "available": True,
        "algorithm_version": "phase2d_c7_video_geometry_v1_reusing_c3c",
        "topology": {key: value for key, value in topology.items() if not isinstance(value, np.ndarray)},
        "ray_diagnostics": ray_diagnostics,
        "water_level_estimation": water_level,
        "seed_diagnostics": reconstruction,
        "water_result": water_result,
        "self_consistency": consistency,
        "prediction_side_quality_gate": gate,
        "quality_status": gate["status"],
        "geometry_diagnostic_readiness": readiness,
        "predicted_dem_mask": predicted_mask,
        "predicted_depth_m": depth,
        "reprojected_camera_mask": reprojected > 127,
    }


def summarize_video_geometry(rows: list[dict[str, Any]], anchor_frame_index: int) -> dict[str, Any]:
    available = [row for row in rows if row["geometry_available"]]
    if not available:
        return {
            "frame_count": len(rows),
            "available_frame_count": 0,
            "quality_status_counts": dict(Counter(row["quality_status"] for row in rows)),
        }

    def stats(field: str) -> dict[str, float]:
        values = np.asarray([float(row[field]) for row in available], dtype=np.float64)
        return {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "coefficient_of_variation": float(np.std(values) / max(abs(np.mean(values)), 1e-12)),
        }

    level_by_frame = {int(row["frame_index"]): float(row["estimated_water_level_m"]) for row in available}
    adjacent_level_changes = [
        abs(level_by_frame[current] - level_by_frame[previous])
        for previous, current in zip(sorted(level_by_frame)[:-1], sorted(level_by_frame)[1:])
        if current == previous + 1
    ]
    anchor = next((row for row in available if int(row["frame_index"]) == anchor_frame_index), None)
    return {
        "frame_count": len(rows),
        "available_frame_count": len(available),
        "failed_frame_count": len(rows) - len(available),
        "estimated_water_level_m": stats("estimated_water_level_m"),
        "water_area_m2": stats("water_area_m2"),
        "water_volume_m3": stats("water_volume_m3"),
        "mean_depth_cm": stats("mean_depth_cm"),
        "max_depth_cm": stats("max_depth_cm"),
        "camera_reprojection_iou": stats("camera_reprojection_iou"),
        "boundary_reprojection_p95_px": stats("boundary_reprojection_p95_px"),
        "adjacent_absolute_water_level_change_m": {
            **stats_from_values(adjacent_level_changes),
            "p95": float(np.percentile(adjacent_level_changes, 95)) if adjacent_level_changes else None,
        },
        "anchor": None if anchor is None else {
            key: anchor[key] for key in (
                "frame_index", "estimated_water_level_m", "water_area_m2", "water_volume_m3",
                "mean_depth_cm", "max_depth_cm", "camera_reprojection_iou",
                "boundary_reprojection_p95_px", "quality_status",
            )
        },
        "quality_status_counts": dict(sorted(Counter(row["quality_status"] for row in rows).items())),
        "gate_reason_counts": dict(sorted(Counter(
            reason for row in rows for reason in row.get("gate_reasons", [])
        ).items())),
        "ground_truth_used": False,
        "eligible_for_downstream": False,
    }


def stats_from_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None, "std": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }
