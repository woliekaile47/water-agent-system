#!/usr/bin/env python3
"""Fixed-method, GT-isolated Phase 2D-B-2B shoreline topology ablation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.evaluation.water_surface_aware_quality_gate import evaluate_water_surface_aware_quality_gate
from src.fusion.project_camera_mask_to_dem import camera_model, project_camera_mask_to_dem
from src.fusion.water_surface_aware_mask_to_dem import (
    camera_ray_map,
    intersect_ray_with_dem,
    reproject_water_surface,
)
from src.hydrology.estimate_water_level_from_boundary import connected_components
from src.hydrology.estimate_water_level_from_shoreline import estimate_water_level_from_shoreline
from src.hydrology.invert_depth_from_ground_dem import invert_depth_from_ground_dem
from src.integration.unknown_aware_geometry import (
    build_trusted_shoreline,
    camera_reprojection_consistency_unknown_aware,
    intersect_trusted_camera_shoreline,
    reconstruct_connected_lowland_unknown_aware,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def _finite(values) -> list[float]:
    return [float(value) for value in values if value is not None and np.isfinite(value)]


def _mean(values) -> float | None:
    finite = _finite(values)
    return None if not finite else float(np.mean(finite))


def _median(values) -> float | None:
    finite = _finite(values)
    return None if not finite else float(np.median(finite))


def boundary_p50_distance_transform(
    observed_water: np.ndarray, unknown: np.ndarray, reprojected_water: np.ndarray,
) -> float | None:
    """Exact Euclidean P50 on the frozen symmetric trusted-boundary domain."""
    observed = np.asarray(observed_water, dtype=bool)
    unknown_mask = np.asarray(unknown, dtype=bool)
    reprojected = np.asarray(reprojected_water, dtype=bool) & ~unknown_mask
    observed_boundary, _, _ = build_trusted_shoreline(observed, unknown_mask)
    reprojected_boundary, _, _ = build_trusted_shoreline(reprojected, unknown_mask)
    if not np.any(observed_boundary) or not np.any(reprojected_boundary):
        return None
    to_reprojected = cv2.distanceTransform(
        (~reprojected_boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
    )[observed_boundary]
    to_observed = cv2.distanceTransform(
        (~observed_boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
    )[reprojected_boundary]
    return float(np.percentile(np.concatenate((to_reprojected, to_observed)), 50))


def component_count(mask: np.ndarray, connectivity: int = 8) -> int:
    if not np.any(mask):
        return 0
    count, _ = cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=connectivity)
    return int(count - 1)


def hole_components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    water = np.asarray(mask, dtype=bool)
    if water.ndim != 2 or not np.any(water):
        return []
    background = ~water
    exterior = np.zeros_like(background)
    queue: list[tuple[int, int]] = []
    height, width = water.shape
    for row in range(height):
        for column in (0, width - 1):
            if background[row, column] and not exterior[row, column]:
                exterior[row, column] = True; queue.append((row, column))
    for column in range(width):
        for row in (0, height - 1):
            if background[row, column] and not exterior[row, column]:
                exterior[row, column] = True; queue.append((row, column))
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1)) if connectivity == 4 else (
        (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)
    )
    cursor = 0
    while cursor < len(queue):
        row, column = queue[cursor]; cursor += 1
        for drow, dcolumn in neighbors:
            nr, nc = row + drow, column + dcolumn
            if 0 <= nr < height and 0 <= nc < width and background[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True; queue.append((nr, nc))
    return connected_components(background & ~exterior, connectivity)


def largest_component(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    components = connected_components(source, connectivity)
    if not components:
        return np.zeros_like(source)
    return np.asarray(max(components, key=np.count_nonzero), dtype=bool).copy()


def filter_small_components(mask: np.ndarray, minimum_area_px: int, connectivity: int = 8) -> np.ndarray:
    result = np.zeros_like(np.asarray(mask, dtype=bool))
    for component in connected_components(np.asarray(mask, dtype=bool), connectivity):
        if int(np.count_nonzero(component)) >= int(minimum_area_px):
            result |= component
    return result


def conditional_hole_fill(
    mask: np.ndarray, max_area_px: int, max_water_area_fraction: float, connectivity: int = 8,
) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    result = source.copy()
    maximum = min(int(max_area_px), int(np.floor(np.count_nonzero(source) * float(max_water_area_fraction))))
    if maximum <= 0:
        return result
    for hole in hole_components(source, connectivity):
        if int(np.count_nonzero(hole)) <= maximum:
            result |= hole
    return result


def apply_topology_method(
    mask: np.ndarray, unknown: np.ndarray, method: str, parameters: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(mask, dtype=bool)
    unknown_mask = np.asarray(unknown, dtype=bool)
    if source.shape != unknown_mask.shape:
        raise ValueError("mask and unknown must have identical shapes")
    connectivity = int(parameters["connectivity"])
    outer_only = method.endswith("outer_boundary_only")
    if method == "baseline" or method == "outer_boundary_only":
        result = source.copy()
    elif method in ("largest_component", "largest_component_outer_boundary_only"):
        result = largest_component(source, connectivity)
    elif method == "small_component_filter":
        result = filter_small_components(source, parameters["small_component_min_area_px"], connectivity)
    elif method == "conditional_hole_fill":
        result = conditional_hole_fill(
            source, parameters["conditional_hole_max_area_px"],
            parameters["conditional_hole_max_water_area_fraction"], connectivity,
        )
    elif method == "morphological_closing":
        size = int(parameters["morphological_closing_kernel_px"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        result = cv2.morphologyEx(
            source.astype(np.uint8), cv2.MORPH_CLOSE, kernel,
            iterations=int(parameters["morphological_closing_iterations"]),
        ) > 0
    elif method == "largest_component_conditional_hole_fill":
        result = largest_component(source, connectivity)
        result = conditional_hole_fill(
            result, parameters["conditional_hole_max_area_px"],
            parameters["conditional_hole_max_water_area_fraction"], connectivity,
        )
    elif method == "small_component_filter_conditional_hole_fill":
        result = filter_small_components(source, parameters["small_component_min_area_px"], connectivity)
        result = conditional_hole_fill(
            result, parameters["conditional_hole_max_area_px"],
            parameters["conditional_hole_max_water_area_fraction"], connectivity,
        )
    else:
        raise ValueError(f"Unsupported fixed topology method: {method}")
    result = np.asarray(result, dtype=bool) & ~unknown_mask
    return result.copy(), {
        "method": method, "outer_boundary_only_for_water_level": outer_only,
        "input_modified_in_place": False, "ground_truth_used": False,
    }


def _outer_interface_points(water: np.ndarray, unknown: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    main = largest_component(water)
    contours, _ = cv2.findContours(main.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    trusted = np.zeros_like(main)
    points: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    known_nonwater = ~water & ~unknown
    height, width = water.shape
    for contour in contours:
        for point in contour[:, 0, :]:
            column, row = int(point[0]), int(point[1])
            for drow, dcolumn in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + drow, column + dcolumn
                key = (row, column, drow, dcolumn)
                if key in seen or not (0 <= nr < height and 0 <= nc < width) or not known_nonwater[nr, nc]:
                    continue
                seen.add(key); trusted[row, column] = True
                points.append({
                    "water_row": row, "water_col": column,
                    "pixel_v": float(row) + 0.5 * drow,
                    "pixel_u": float(column) + 0.5 * dcolumn,
                })
    return trusted, points


def intersect_outer_shoreline(
    water: np.ndarray, unknown: np.ndarray, ground_dem: np.ndarray,
    sensors: dict[str, Any], mapping: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    trusted, points = _outer_interface_points(water, unknown)
    stride = max(1, int(mapping.get("shoreline_sample_stride_px", 2)))
    model = camera_model(sensors)
    records, failures = [], Counter()
    for point in points[::stride]:
        origin, direction = camera_ray_map(point["pixel_u"], point["pixel_v"], model)
        hit, reason = intersect_ray_with_dem(origin, direction, ground_dem, sensors, mapping)
        if hit is None:
            failures[reason] += 1
            continue
        records.append({
            "component_index": 0, "pixel_u": point["pixel_u"], "pixel_v": point["pixel_v"],
            "ray_direction_map": direction.tolist(), **hit,
        })
    rows, columns = np.where(water)
    edge_band = max(1, int(mapping.get("image_edge_band_px", 2)))
    edge = ((rows < edge_band) | (rows >= water.shape[0] - edge_band)
            | (columns < edge_band) | (columns >= water.shape[1] - edge_band))
    sampled = len(points[::stride])
    return trusted, records, {
        "water_component_count": component_count(water), "retained_water_component_count": 1,
        "candidate_interface_count": len(points), "trusted_interface_count": len(points),
        "unknown_adjacent_interface_count": 0, "image_edge_interface_count": 0,
        "trusted_shoreline_fraction": 1.0 if points else 0.0,
        "sampled_shoreline_ray_count": sampled, "successful_intersection_count": len(records),
        "shoreline_intersection_success_rate": float(len(records) / max(1, sampled)),
        "intersection_failure_reasons": dict(sorted(failures.items())),
        "camera_mask_edge_touch_ratio": float(np.count_nonzero(edge) / max(1, rows.size)),
        "camera_frame": model["frame_id"], "map_frame": model["map_frame_id"],
        "transform": "T_map_camera_optical", "shoreline_semantics": "largest_component_external_contour_only",
        "ground_truth_used": False,
    }


def run_geometry_from_existing_mask(
    repaired_water: np.ndarray, unknown: np.ndarray, outer_boundary_only: bool,
    ground_dem: np.ndarray, sensors: dict[str, Any], mapping: dict[str, Any], gate_config: dict[str, Any],
) -> dict[str, Any]:
    water = np.asarray(repaired_water, dtype=bool)
    unknown_mask = np.asarray(unknown, dtype=bool)
    if not np.any(water):
        return {"available": False, "failure_type": "empty_repaired_camera_water_mask", "gate": {
            "status": "reject", "reasons": ["empty_repaired_camera_water_mask"], "eligible_for_downstream": False,
        }}
    seed_mask, seed_diagnostics = project_camera_mask_to_dem(
        ground_dem, np.where(water, 255, 0).astype(np.uint8), sensors, int(mapping.get("mask_threshold", 127)),
    )
    if not np.any(seed_mask):
        return {"available": False, "failure_type": "no_valid_seed", "gate": {
            "status": "reject", "reasons": ["no_valid_seed"], "eligible_for_downstream": False,
        }}
    if outer_boundary_only:
        trusted, intersections, ray_diagnostics = intersect_outer_shoreline(
            water, unknown_mask, ground_dem, sensors, mapping,
        )
    else:
        trusted, intersections, ray_diagnostics = intersect_trusted_camera_shoreline(
            water, unknown_mask, ground_dem, sensors, mapping,
        )
    if not intersections:
        return {"available": False, "failure_type": "no_valid_shoreline_intersections", "gate": {
            "status": "reject", "reasons": ["no_valid_shoreline_intersections"], "eligible_for_downstream": False,
        }}
    try:
        water_level, shoreline_diagnostics = estimate_water_level_from_shoreline(
            intersections, mapping["shoreline_water_level"],
        )
    except ValueError as error:
        return {"available": False, "failure_type": "water_level_estimation_failed", "message": str(error), "gate": {
            "status": "reject", "reasons": ["water_level_estimation_failed"], "eligible_for_downstream": False,
        }}
    shoreline_diagnostics["estimated_water_level_m"] = float(water_level)
    predicted_mask, reconstruction = reconstruct_connected_lowland_unknown_aware(
        ground_dem, water_level, seed_mask, mapping["reconstruction"], water, unknown_mask, sensors,
    )
    reconstruction["initial_seed_projection"] = seed_diagnostics
    if not reconstruction["seed_valid"] or not np.any(predicted_mask):
        return {"available": False, "failure_type": "invalid_or_empty_reconstruction", "gate": {
            "status": "reject", "reasons": ["invalid_or_empty_reconstruction"], "eligible_for_downstream": False,
        }, "reconstruction": reconstruction}
    cell_size = float(sensors["road"]["dem_resolution_m"])
    depth, predicted_mask, water_result = invert_depth_from_ground_dem(
        ground_dem, predicted_mask, water_level, cell_size,
    )
    reprojected, projection = reproject_water_surface(predicted_mask, water_level, sensors)
    consistency = camera_reprojection_consistency_unknown_aware(water, unknown_mask, reprojected, projection)
    consistency["boundary_reprojection_p50_px"] = boundary_p50_distance_transform(
        water, unknown_mask, reprojected > 127,
    )
    gate = evaluate_water_surface_aware_quality_gate(
        ray_diagnostics, shoreline_diagnostics, reconstruction, consistency,
        water_result, depth, gate_config, required_files=None,
    )
    return {
        "available": True, "water_mask": water.copy(), "trusted_shoreline_mask": trusted,
        "water_level_m": float(water_level), "predicted_dem_mask": predicted_mask,
        "predicted_depth_m": depth, "reprojected_camera_mask": reprojected > 127,
        "ray_diagnostics": ray_diagnostics, "shoreline_diagnostics": shoreline_diagnostics,
        "reconstruction": reconstruction, "water_result": water_result,
        "self_consistency": consistency, "gate": gate,
        "ground_truth_used": False, "temporal_prediction_rerun": False,
        "eligible_for_downstream": False,
    }


def run_fixed_candidates_before_gt(
    source_water: np.ndarray, unknown: np.ndarray, methods: list[str], parameters: dict[str, Any],
    ground_dem: np.ndarray, sensors: dict[str, Any], mapping: dict[str, Any], gate_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Complete every fixed candidate before any evaluation function is allowed to load GT."""
    candidates = {}
    for method in methods:
        repaired, transform = apply_topology_method(source_water, unknown, method, parameters)
        geometry = run_geometry_from_existing_mask(
            repaired, unknown, transform["outer_boundary_only_for_water_level"],
            ground_dem, sensors, mapping, gate_config,
        )
        candidates[method] = {"repaired_water_mask": repaired, "transform": transform, "geometry": geometry}
    return candidates


def _mask_iou(predicted: np.ndarray, truth: np.ndarray) -> float:
    predicted_mask, truth_mask = np.asarray(predicted, dtype=bool), np.asarray(truth, dtype=bool)
    return float(np.count_nonzero(predicted_mask & truth_mask) / max(1, np.count_nonzero(predicted_mask | truth_mask)))


def _depth_mae(predicted: np.ndarray, truth: np.ndarray, truth_mask: np.ndarray) -> float | None:
    valid = np.asarray(truth_mask, dtype=bool) & np.isfinite(predicted) & np.isfinite(truth)
    return None if not np.any(valid) else float(np.mean(np.abs(predicted[valid] - truth[valid])))


def unobservable_component_safety(
    component: np.ndarray, projected_component: np.ndarray,
    repaired_camera_mask: np.ndarray, predicted_dem_mask: np.ndarray,
) -> dict[str, Any]:
    projected = np.asarray(projected_component, dtype=bool)
    camera_overlap = int(np.count_nonzero(projected & np.asarray(repaired_camera_mask, dtype=bool)))
    dem_overlap = int(np.count_nonzero(np.asarray(component, dtype=bool) & np.asarray(predicted_dem_mask, dtype=bool)))
    projected_pixels = int(np.count_nonzero(projected))
    return {
        "camera_projected_pixels": projected_pixels,
        "repaired_camera_evidence_overlap_pixels": camera_overlap,
        "predicted_dem_intersection_cells": dem_overlap,
        "unobservable_safe": bool(projected_pixels == 0 and camera_overlap == 0 and dem_overlap == 0),
    }


def evaluate_candidates_after_prediction(
    sequence_dir: str | Path, case_id: str, rain_level: str, seed: int,
    candidates: dict[str, dict[str, Any]], source_water: np.ndarray,
    existing_geometry_record: dict[str, Any], project_root: str | Path, sensors: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # GT imports and reads intentionally occur only after all candidate geometry is complete.
    from src.evaluation.evaluate_simulation_depth import load_ground_truth_evaluation_inputs

    sequence = Path(sequence_dir).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    camera_gt = _read_mask(sequence / "ground_truth/water_mask.png")
    static_gt = load_ground_truth_evaluation_inputs(root, case_id)
    baseline = candidates["baseline"]
    if not baseline["geometry"]["available"]:
        raise RuntimeError(f"Baseline geometry unavailable for {case_id}/{rain_level}/seed_{seed}")
    baseline_geometry = baseline["geometry"]
    before = {
        "component_count": component_count(source_water),
        "hole_count": len(hole_components(source_water)),
        "area": int(np.count_nonzero(source_water)),
        "camera_iou": _mask_iou(source_water, camera_gt),
        "p50": baseline_geometry["self_consistency"]["boundary_reprojection_p50_px"],
        "p95": baseline_geometry["self_consistency"]["boundary_reprojection_p95_px"],
        "reprojection_iou": baseline_geometry["self_consistency"]["camera_reprojection_iou"],
        "water_level_error": abs(baseline_geometry["water_level_m"] - float(static_gt["water_level_m"])),
        "depth_mae": _depth_mae(
            baseline_geometry["predicted_depth_m"], np.asarray(static_gt["depth_map"], dtype=np.float32),
            np.asarray(static_gt["dem_mask"], dtype=bool),
        ),
        "existing_p95": existing_geometry_record["boundary_reprojection_p95_px"],
    }
    rows, safety = [], []
    for method, candidate in candidates.items():
        repaired, geometry = candidate["repaired_water_mask"], candidate["geometry"]
        area_after = int(np.count_nonzero(repaired))
        row = {
            "data_role": "independent_topology_ablation_evaluation",
            "case_id": case_id, "nominal_depth_cm": int(case_id.split("_")[-2].removesuffix("cm")),
            "rain_level": rain_level, "seed": seed, "method": method,
            "observed_component_count_before": before["component_count"],
            "observed_component_count_after": component_count(repaired),
            "hole_count_before": before["hole_count"], "hole_count_after": len(hole_components(repaired)),
            "mask_area_before_px": before["area"], "mask_area_after_px": area_after,
            "mask_area_change_fraction": float((area_after - before["area"]) / max(1, before["area"])),
            "camera_mask_iou_before": before["camera_iou"], "camera_mask_iou_after": _mask_iou(repaired, camera_gt),
            "boundary_reprojection_p50_before_px": before["p50"],
            "boundary_reprojection_p95_before_px": before["p95"],
            "baseline_existing_p95_absolute_delta_px": abs(before["p95"] - before["existing_p95"]),
            "camera_reprojection_iou_before": before["reprojection_iou"],
            "water_level_absolute_error_before_m": before["water_level_error"],
            "depth_mae_before_m": before["depth_mae"],
            "geometry_available": bool(geometry["available"]),
            "geometry_gate_status_under_existing_thresholds": geometry["gate"]["status"],
            "geometry_reject_reasons_under_existing_thresholds": geometry["gate"].get("reasons", []),
            "parameters_fixed_before_gt_evaluation": True,
            "per_case_method_selection": False, "authoritative_measurement": False,
            "eligible_for_downstream": False,
        }
        if geometry["available"]:
            row.update({
                "boundary_reprojection_p50_after_px": geometry["self_consistency"]["boundary_reprojection_p50_px"],
                "boundary_reprojection_p95_after_px": geometry["self_consistency"]["boundary_reprojection_p95_px"],
                "camera_reprojection_iou_after": geometry["self_consistency"]["camera_reprojection_iou"],
                "water_level_absolute_error_after_m": abs(geometry["water_level_m"] - float(static_gt["water_level_m"])),
                "depth_mae_after_m": _depth_mae(
                    geometry["predicted_depth_m"], np.asarray(static_gt["depth_map"], dtype=np.float32),
                    np.asarray(static_gt["dem_mask"], dtype=bool),
                ),
            })
        else:
            row.update({key: None for key in (
                "boundary_reprojection_p50_after_px", "boundary_reprojection_p95_after_px",
                "camera_reprojection_iou_after", "water_level_absolute_error_after_m", "depth_mae_after_m",
            )})
        rows.append(row)
        if case_id == "sim_water_40cm_001" and geometry["available"]:
            gt_components = connected_components(np.asarray(static_gt["dem_mask"], dtype=bool), 8)
            component_rows = []
            for index, component in enumerate(gt_components):
                projected, _ = reproject_water_surface(component, float(static_gt["water_level_m"]), sensors)
                projected_mask = projected > 127
                projected_pixels = int(np.count_nonzero(projected_mask))
                component_rows.append({
                    "component_index": index, "ground_truth_cell_count": int(np.count_nonzero(component)),
                    **unobservable_component_safety(
                        component, projected_mask, repaired, geometry["predicted_dem_mask"],
                    ),
                })
            invisible = [item for item in component_rows if item["camera_projected_pixels"] == 0]
            safety.append({
                "case_id": case_id, "rain_level": rain_level, "seed": seed, "method": method,
                "components": component_rows,
                "unobservable_secondary_basin_count": len(invisible),
                "unobservable_secondary_camera_evidence_pixels": sum(item["repaired_camera_evidence_overlap_pixels"] for item in invisible),
                "unobservable_secondary_predicted_dem_cells": sum(item["predicted_dem_intersection_cells"] for item in invisible),
                "unobservable_secondary_basin_safe": bool(invisible) and all(
                    item["repaired_camera_evidence_overlap_pixels"] == 0
                    and item["predicted_dem_intersection_cells"] == 0 for item in invisible
                ),
                "ground_truth_role": "independent_safety_evaluation_only",
                "eligible_for_downstream": False,
            })
    return rows, safety


def evaluate_dry_safety(
    case_id: str, rain_level: str, seed: int, source_water: np.ndarray,
    unknown: np.ndarray, methods: list[str], parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        repaired, _ = apply_topology_method(source_water, unknown, method, parameters)
        rows.append({
            "case_id": case_id, "rain_level": rain_level, "seed": seed, "method": method,
            "input_water_pixels": int(np.count_nonzero(source_water)),
            "output_water_pixels": int(np.count_nonzero(repaired)),
            "empty_preserved": bool(not np.any(source_water) and not np.any(repaired)),
        })
    return rows


def build_method_summary(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    extreme_keys = {(row["case_id"], row["rain_level"], row["seed"]) for row in rows if row["method"] == "baseline" and row["boundary_reprojection_p95_before_px"] > 10}
    near_keys = {(row["case_id"], row["rain_level"], row["seed"]) for row in rows if row["method"] == "baseline" and 3 < row["boundary_reprojection_p95_before_px"] <= 4}
    high_iou_keys = {(row["case_id"], row["rain_level"], row["seed"]) for row in rows if row["method"] == "baseline" and row["camera_mask_iou_before"] >= .8}
    summaries = {}
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        available = [row for row in group if row["geometry_available"]]
        p95_delta = [row["boundary_reprojection_p95_after_px"] - row["boundary_reprojection_p95_before_px"] for row in available]
        camera_delta = [row["camera_mask_iou_after"] - row["camera_mask_iou_before"] for row in group]
        area_delta = [row["mask_area_change_fraction"] for row in group]
        def keyed(keys):
            return [row for row in available if (row["case_id"], row["rain_level"], row["seed"]) in keys]
        summaries[method] = {
            "sequence_count": len(group), "geometry_available_count": len(available),
            "p95_change_mean_px": _mean(p95_delta), "p95_change_median_px": _median(p95_delta),
            "p95_improved_count": sum(delta < 0 for delta in p95_delta),
            "p95_worsened_count": sum(delta > 0 for delta in p95_delta),
            "p95_unchanged_count": sum(abs(delta) <= 1e-12 for delta in p95_delta),
            "camera_iou_change_mean": _mean(camera_delta), "camera_iou_decreased_count": sum(delta < 0 for delta in camera_delta),
            "area_change_mean_fraction": _mean(area_delta), "area_change_max_fraction": max(_finite(area_delta), default=None),
            "area_change_min_fraction": min(_finite(area_delta), default=None),
            "existing_geometry_gate_pass_count": sum(row["geometry_gate_status_under_existing_thresholds"] == "pass" for row in group),
            "extreme_p95_change_mean_px": _mean(row["boundary_reprojection_p95_after_px"] - row["boundary_reprojection_p95_before_px"] for row in keyed(extreme_keys)),
            "extreme_p95_improved_count": sum(row["boundary_reprojection_p95_after_px"] < row["boundary_reprojection_p95_before_px"] for row in keyed(extreme_keys)),
            "near_threshold_p95_change_mean_px": _mean(row["boundary_reprojection_p95_after_px"] - row["boundary_reprojection_p95_before_px"] for row in keyed(near_keys)),
            "near_threshold_worsened_count": sum(row["boundary_reprojection_p95_after_px"] > row["boundary_reprojection_p95_before_px"] for row in keyed(near_keys)),
            "high_camera_iou_p95_change_mean_px": _mean(row["boundary_reprojection_p95_after_px"] - row["boundary_reprojection_p95_before_px"] for row in keyed(high_iou_keys)),
            "high_camera_iou_improved_count": sum(row["boundary_reprojection_p95_after_px"] < row["boundary_reprojection_p95_before_px"] for row in keyed(high_iou_keys)),
        }
    return {
        "data_role": "fixed_method_ablation_summary", "water_sequence_count": len(rows) // len(methods),
        "method_count": len(methods), "methods": summaries,
        "parameters_fixed_before_gt_evaluation": True, "per_case_best_method_selected": False,
        "thresholds_modified": False, "temporal_prediction_rerun": False,
        "authoritative_measurement": False, "eligible_for_downstream": False,
        "baseline_existing_p95_max_absolute_delta_px": max(
            _finite(row.get("baseline_existing_p95_absolute_delta_px") for row in rows), default=None,
        ),
    }


CSV_FIELDS = [
    "case_id", "nominal_depth_cm", "rain_level", "seed", "method",
    "observed_component_count_before", "observed_component_count_after", "hole_count_before", "hole_count_after",
    "mask_area_before_px", "mask_area_after_px", "mask_area_change_fraction",
    "camera_mask_iou_before", "camera_mask_iou_after",
    "boundary_reprojection_p50_before_px", "boundary_reprojection_p50_after_px",
    "boundary_reprojection_p95_before_px", "boundary_reprojection_p95_after_px",
    "camera_reprojection_iou_before", "camera_reprojection_iou_after",
    "water_level_absolute_error_before_m", "water_level_absolute_error_after_m",
    "depth_mae_before_m", "depth_mae_after_m", "geometry_available",
    "geometry_gate_status_under_existing_thresholds", "geometry_reject_reasons_under_existing_thresholds",
]


def write_outputs(
    output_root: str | Path, rows: list[dict[str, Any]], methods: list[str],
    dry_rows: list[dict[str, Any]], safety_rows: list[dict[str, Any]], config_snapshot: dict[str, Any],
) -> dict[str, Any]:
    output = Path(output_root).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    method_summary = build_method_summary(rows, methods)
    dry_summary = {
        "sequence_count": len(dry_rows) // len(methods), "method_count": len(methods),
        "all_empty_preserved": all(row["empty_preserved"] for row in dry_rows), "rows": dry_rows,
    }
    forty_summary = {
        "row_count": len(safety_rows),
        "all_methods_preserve_unobservable_secondary_basin": all(row["unobservable_secondary_basin_safe"] for row in safety_rows),
        "rows": safety_rows,
    }
    documents = {
        "topology_ablation_results.json": rows, "method_summary.json": method_summary,
        "dry_safety_summary.json": dry_summary, "forty_cm_basin_safety.json": forty_summary,
        "experiment_config_snapshot.json": config_snapshot,
    }
    for name, document in documents.items():
        (output / name).write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "topology_ablation_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS); writer.writeheader()
        for item in rows:
            row = {field: item.get(field) for field in CSV_FIELDS}
            row["geometry_reject_reasons_under_existing_thresholds"] = ";".join(item["geometry_reject_reasons_under_existing_thresholds"])
            writer.writerow(row)
    write_figures_compatible(output)
    return method_summary


def write_figures_compatible(output: Path) -> None:
    if int(np.__version__.split(".", 1)[0]) >= 2 and os.environ.get("PHASE2DB2B_MPL_COMPAT") != "1":
        environment = dict(os.environ); environment.update({"PYTHONNOUSERSITE": "1", "PHASE2DB2B_MPL_COMPAT": "1"})
        subprocess.run(["/usr/bin/python3", "-m", "src.evaluation.evaluate_shoreline_topology_ablation", "--render-existing", str(output)], check=True, env=environment)
        return
    write_figures(output)


def write_figures(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = _read_json(output / "topology_ablation_results.json")
    methods = list(_read_json(output / "method_summary.json")["methods"])
    short = {method: method.replace("_component", "_comp").replace("conditional_hole_fill", "hole_fill").replace("morphological_closing", "closing") for method in methods}

    def paired_bar(filename, before_key, after_key, ylabel, title):
        before = [_mean(row[before_key] for row in rows if row["method"] == method) for method in methods]
        after = [_mean(row[after_key] for row in rows if row["method"] == method) for method in methods]
        x = np.arange(len(methods)); figure, axis = plt.subplots(figsize=(12, 5))
        axis.bar(x - .2, before, .4, label="before"); axis.bar(x + .2, after, .4, label="after")
        axis.set_xticks(x, [short[item] for item in methods], rotation=25, ha="right")
        axis.set_ylabel(ylabel); axis.set_title(title); axis.legend(); figure.tight_layout(); figure.savefig(output / filename, dpi=150); plt.close(figure)
    paired_bar("p95_before_after_by_method.png", "boundary_reprojection_p95_before_px", "boundary_reprojection_p95_after_px", "Mean P95 (px)", "Boundary P95 before/after by fixed method")
    paired_bar("camera_iou_before_after_by_method.png", "camera_mask_iou_before", "camera_mask_iou_after", "Mean Camera IoU", "Camera IoU before/after by fixed method")
    figure, axis = plt.subplots(figsize=(12, 5)); data = [[row["mask_area_change_fraction"] for row in rows if row["method"] == method] for method in methods]
    axis.boxplot(data, labels=[short[item] for item in methods]); axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("Mask area change fraction"); axis.set_title("Area change by fixed method"); axis.tick_params(axis="x", rotation=25); figure.tight_layout(); figure.savefig(output / "area_change_by_method.png", dpi=150); plt.close(figure)

    def comparison(filename, predicate, title):
        baseline_rows = [row for row in rows if row["method"] == "baseline" and predicate(row)]
        keys = [(row["case_id"], row["rain_level"], row["seed"]) for row in baseline_rows]
        matrix = np.asarray([[next((row["boundary_reprojection_p95_after_px"] for row in rows if row["method"] == method and (row["case_id"], row["rain_level"], row["seed"]) == key), np.nan) for method in methods] for key in keys])
        figure, axis = plt.subplots(figsize=(12, max(4, .7 * len(keys)))); image = axis.imshow(matrix, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(methods)), [short[item] for item in methods], rotation=25, ha="right")
        axis.set_yticks(range(len(keys)), [f"{key[0].split('_')[2]}-{key[1]}-{key[2]}" for key in keys]); axis.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, f"{matrix[i,j]:.1f}" if np.isfinite(matrix[i,j]) else "NA", ha="center", va="center", color="white", fontsize=8)
        figure.colorbar(image, ax=axis, label="Boundary P95 (px)"); figure.tight_layout(); figure.savefig(output / filename, dpi=150); plt.close(figure)
    comparison("extreme_case_comparison.png", lambda row: row["boundary_reprojection_p95_before_px"] > 10, "Extreme cases: P95 by fixed method")
    comparison("near_threshold_case_comparison.png", lambda row: 3 < row["boundary_reprojection_p95_before_px"] <= 4, "Near-threshold cases: P95 by fixed method")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--render-existing"); args = parser.parse_args()
    if not args.render_existing: parser.error("--render-existing is required when invoking this module directly")
    write_figures(Path(args.render_existing).expanduser().resolve())


if __name__ == "__main__":
    main()
