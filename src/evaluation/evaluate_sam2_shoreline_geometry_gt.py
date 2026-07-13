#!/usr/bin/env python3
"""Independent Ground Truth evaluation for Phase 2D-C-3C outputs.

This is the only Phase 2D-C module allowed to load water-state Ground Truth.
It never writes to, reselects, or recomputes the saved prediction artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.fusion.sam2_shoreline_geometry_adapter import outer_boundary_mask
from src.fusion.water_surface_aware_mask_to_dem import (
    _nearest_boundary_distances,
    intersect_camera_shoreline,
)
from src.hydrology.estimate_water_level_from_boundary import connected_components
from src.hydrology.estimate_water_level_from_shoreline import estimate_water_level_from_shoreline
from src.hydrology.invert_boundary_waterline_depth import extract_boundary_mask


def load_ground_truth_evaluation_inputs(
    project_root: str | Path,
    case_id: str,
    sequence_relative_path: str | Path,
) -> dict[str, Any]:
    """Load and validate water-state answer data for evaluation only."""
    root = Path(project_root).expanduser().resolve()
    case_dir = root / "data" / "simulation" / case_id
    sequence_dir = root / sequence_relative_path
    paths = {
        "case_manifest": case_dir / "manifest.json",
        "sequence_manifest": sequence_dir / "metadata" / "sequence_manifest.json",
        "camera_mask": case_dir / "ground_truth" / "camera_water_mask_gt.png",
        "sequence_camera_mask": sequence_dir / "ground_truth" / "water_mask.png",
        "dem_mask": case_dir / "ground_truth" / "dem_water_mask_gt.npy",
        "depth_map": case_dir / "ground_truth" / "depth_map_gt_m.npy",
        "ground_dem": case_dir / "ground_truth" / "ground_dem_gt.npy",
        "metadata": case_dir / "ground_truth" / "ground_truth_metadata.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Ground Truth evaluation inputs: {missing}")
    case_manifest = json.loads(paths["case_manifest"].read_text(encoding="utf-8"))
    sequence_manifest = json.loads(paths["sequence_manifest"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if case_manifest["case_id"] != case_id or sequence_manifest["case_id"] != case_id:
        raise ValueError("Case and sequence manifests do not identify the same case")
    camera_mask = np.asarray(Image.open(paths["camera_mask"]).convert("L")) > 127
    sequence_mask = np.asarray(Image.open(paths["sequence_camera_mask"]).convert("L")) > 127
    if camera_mask.shape != (360, 640) or not np.array_equal(camera_mask, sequence_mask):
        raise ValueError("Camera Ground Truth is not a strict static-mask match for the sequence")
    dem_mask = np.load(paths["dem_mask"], allow_pickle=False).astype(bool)
    depth_map = np.load(paths["depth_map"], allow_pickle=False).astype(np.float32)
    ground_dem = np.load(paths["ground_dem"], allow_pickle=False).astype(np.float32)
    if dem_mask.shape != depth_map.shape or dem_mask.shape != ground_dem.shape:
        raise ValueError("DEM Ground Truth shapes differ")
    if not np.isfinite(depth_map).all() or np.any(depth_map < 0.0) or np.any(depth_map[~dem_mask] != 0.0):
        raise ValueError("Depth Ground Truth violates finite/nonnegative/mask constraints")
    resolution = float(case_manifest["config_snapshot"]["sensors"]["road"]["dem_resolution_m"])
    computed_area = float(np.count_nonzero(dem_mask) * resolution * resolution)
    computed_volume = float(np.sum(depth_map.astype(np.float64)) * resolution * resolution)
    if not (
        np.isclose(computed_area, float(case_manifest["water_area_m2"]))
        and np.isclose(computed_area, float(metadata["water_area_m2"]))
        and np.isclose(computed_volume, float(case_manifest["water_volume_m3"]))
        and np.isclose(computed_volume, float(metadata["water_volume_m3"]))
    ):
        raise ValueError("Ground Truth area or volume does not match numerical integration")
    validation = {
        "data_role": "independent_evaluation_ground_truth",
        "case_id": case_id,
        "case_sequence_match": True,
        "static_water_state_applies_to_all_sequence_frames": True,
        "camera_mask_shape_hw": list(camera_mask.shape),
        "camera_water_pixel_count": int(np.count_nonzero(camera_mask)),
        "camera_mask_sequence_equal": True,
        "dem_shape_rc": list(dem_mask.shape),
        "dem_water_cell_count": int(np.count_nonzero(dem_mask)),
        "depth_nonnegative": True,
        "depth_outside_mask_zero": True,
        "area_numerically_consistent": True,
        "volume_numerically_consistent": True,
        "paths": {key: str(path) for key, path in paths.items()},
    }
    return {
        "camera_mask": camera_mask,
        "dem_mask": dem_mask,
        "depth_map": depth_map,
        "ground_dem": ground_dem,
        "water_level_m": float(case_manifest["water_level_m"]),
        "water_area_m2": float(case_manifest["water_area_m2"]),
        "water_volume_m3": float(case_manifest["water_volume_m3"]),
        "nominal_depth_m": float(case_manifest["water_depth_cm"]) / 100.0,
        "case_manifest": case_manifest,
        "sequence_manifest": sequence_manifest,
        "validation": validation,
    }


def binary_mask_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(predicted, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("Mask shapes differ")
    intersection = int(np.count_nonzero(prediction & target))
    union = int(np.count_nonzero(prediction | target))
    predicted_count = int(np.count_nonzero(prediction))
    target_count = int(np.count_nonzero(target))
    precision = float(intersection / max(1, predicted_count))
    recall = float(intersection / max(1, target_count))
    return {
        "predicted_pixels": predicted_count,
        "gt_pixels": target_count,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": float(intersection / max(1, union)),
        "precision": precision,
        "recall": recall,
        "f1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
        "false_positive_pixels": int(np.count_nonzero(prediction & ~target)),
        "false_negative_pixels": int(np.count_nonzero(~prediction & target)),
        "area_pixel_absolute_error": abs(predicted_count - target_count),
        "area_pixel_relative_error": float(abs(predicted_count - target_count) / max(1, target_count)),
        "connected_component_count": len(connected_components(prediction, 8)),
    }


def _directional_stats(source_boundary: np.ndarray, target_boundary: np.ndarray) -> dict[str, Any]:
    distances = _nearest_boundary_distances(source_boundary, target_boundary)
    if not distances.size:
        return {"count": 0, "p50_px": None, "p95_px": None, "mean_px": None, "max_px": None}
    return {
        "count": int(distances.size),
        "p50_px": float(np.percentile(distances, 50)),
        "p95_px": float(np.percentile(distances, 95)),
        "mean_px": float(np.mean(distances)),
        "max_px": float(np.max(distances)),
    }


def boundary_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(predicted, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    prediction_boundary = extract_boundary_mask(prediction)
    target_boundary = extract_boundary_mask(target)
    prediction_outer = outer_boundary_mask(prediction)
    target_outer = outer_boundary_mask(target)
    forward = _directional_stats(prediction_boundary, target_boundary)
    reverse = _directional_stats(target_boundary, prediction_boundary)
    outer_forward = _directional_stats(prediction_outer, target_outer)
    outer_reverse = _directional_stats(target_outer, prediction_outer)

    def symmetric(first: dict[str, Any], second: dict[str, Any], first_mask: np.ndarray, second_mask: np.ndarray) -> dict[str, Any]:
        a = _nearest_boundary_distances(first_mask, second_mask)
        b = _nearest_boundary_distances(second_mask, first_mask)
        values = np.concatenate((a, b)) if a.size and b.size else np.asarray([], dtype=np.float64)
        return {
            "count": int(values.size),
            "p50_px": float(np.percentile(values, 50)) if values.size else None,
            "p95_px": float(np.percentile(values, 95)) if values.size else None,
            "mean_px": float(np.mean(values)) if values.size else None,
            "max_px": float(np.max(values)) if values.size else None,
        }

    return {
        "boundary_definition": "water pixels adjacent to non-water using existing Phase 2B extraction",
        "outer_boundary_definition": "water pixels adjacent to exterior background; enclosed holes excluded",
        "symmetric_boundary": symmetric(forward, reverse, prediction_boundary, target_boundary),
        "symmetric_outer_boundary": symmetric(outer_forward, outer_reverse, prediction_outer, target_outer),
        "predicted_shoreline_to_gt_boundary": outer_forward,
        "gt_boundary_to_predicted_shoreline": outer_reverse,
        "predicted_boundary_pixel_count": int(np.count_nonzero(prediction_boundary)),
        "gt_boundary_pixel_count": int(np.count_nonzero(target_boundary)),
        "predicted_outer_boundary_pixel_count": int(np.count_nonzero(prediction_outer)),
        "gt_outer_boundary_pixel_count": int(np.count_nonzero(target_outer)),
    }


def water_level_metrics(
    estimated_level_m: float,
    true_level_m: float,
    shoreline_heights_m: np.ndarray,
    nominal_depth_m: float,
) -> dict[str, Any]:
    heights = np.asarray(shoreline_heights_m, dtype=np.float64)
    heights = heights[np.isfinite(heights)]
    signed = float(estimated_level_m - true_level_m)
    absolute = abs(signed)
    near = {
        f"within_{int(round(tolerance * 100))}cm_count": int(np.count_nonzero(np.abs(heights - true_level_m) <= tolerance))
        for tolerance in (0.01, 0.02, 0.05)
    }
    near.update({
        f"within_{int(round(tolerance * 100))}cm_ratio": float(np.mean(np.abs(heights - true_level_m) <= tolerance))
        if heights.size else 0.0
        for tolerance in (0.01, 0.02, 0.05)
    })
    return {
        "estimated_water_level_m": float(estimated_level_m),
        "gt_water_level_m": float(true_level_m),
        "signed_water_level_error_m": signed,
        "absolute_water_level_error_m": absolute,
        "absolute_water_level_error_cm": float(absolute * 100.0),
        "relative_water_level_error": None,
        "relative_water_level_error_note": "Not defined against signed map-datum elevation.",
        "absolute_error_relative_to_nominal_depth": float(absolute / nominal_depth_m),
        "gt_level_percentile_in_predicted_shoreline_heights": float(100.0 * np.mean(heights <= true_level_m))
        if heights.size else None,
        "shoreline_height_sample_count": int(heights.size),
        **near,
    }


def _error_stats(error: np.ndarray, domain: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(domain, dtype=bool) & np.isfinite(error)
    values = np.asarray(error, dtype=np.float64)[valid]
    if not values.size:
        return {
            "cell_count": 0, "mae_m": None, "rmse_m": None, "median_absolute_error_m": None,
            "max_absolute_error_m": None, "bias_m": None,
        }
    absolute = np.abs(values)
    return {
        "cell_count": int(values.size),
        "mae_m": float(np.mean(absolute)),
        "rmse_m": float(np.sqrt(np.mean(values * values))),
        "median_absolute_error_m": float(np.median(absolute)),
        "max_absolute_error_m": float(np.max(absolute)),
        "bias_m": float(np.mean(values)),
    }


def depth_metrics(
    predicted_depth: np.ndarray,
    true_depth: np.ndarray,
    predicted_mask: np.ndarray,
    true_mask: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(predicted_depth, dtype=np.float64)
    target = np.asarray(true_depth, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("Depth map shapes differ")
    predicted_region = np.asarray(predicted_mask, dtype=bool)
    target_region = np.asarray(true_mask, dtype=bool)
    finite = np.isfinite(prediction) & np.isfinite(target)
    error = prediction - target
    return {
        "error_definition": "predicted_depth_m - gt_depth_m",
        "full_valid_dem_domain": _error_stats(error, finite),
        "predicted_region_domain": _error_stats(error, finite & predicted_region),
        "gt_region_domain": _error_stats(error, finite & target_region),
        "overlap_region_domain": _error_stats(error, finite & predicted_region & target_region),
    }


def area_volume_metrics(
    predicted_water_result: dict[str, Any],
    true_area_m2: float,
    true_volume_m3: float,
    true_depth: np.ndarray,
    true_mask: np.ndarray,
) -> dict[str, Any]:
    predicted_area = float(predicted_water_result["water_area_m2"])
    predicted_volume = float(predicted_water_result["water_volume_m3"])
    true_values = np.asarray(true_depth, dtype=np.float64)[np.asarray(true_mask, dtype=bool)]
    true_mean_cm = float(np.mean(true_values) * 100.0) if true_values.size else 0.0
    true_median_cm = float(np.median(true_values) * 100.0) if true_values.size else 0.0
    true_max_cm = float(np.max(true_values) * 100.0) if true_values.size else 0.0
    return {
        "predicted_area_m2": predicted_area,
        "gt_area_m2": float(true_area_m2),
        "area_absolute_error_m2": abs(predicted_area - true_area_m2),
        "area_relative_error": float(abs(predicted_area - true_area_m2) / true_area_m2),
        "predicted_volume_m3": predicted_volume,
        "gt_volume_m3": float(true_volume_m3),
        "volume_absolute_error_m3": abs(predicted_volume - true_volume_m3),
        "volume_relative_error": float(abs(predicted_volume - true_volume_m3) / true_volume_m3),
        "predicted_mean_depth_cm": float(predicted_water_result["mean_depth_cm"]),
        "gt_mean_depth_cm": true_mean_cm,
        "mean_depth_error_cm": float(predicted_water_result["mean_depth_cm"] - true_mean_cm),
        "predicted_median_depth_cm": float(predicted_water_result["median_depth_cm"]),
        "gt_median_depth_cm": true_median_cm,
        "median_depth_error_cm": float(predicted_water_result["median_depth_cm"] - true_median_cm),
        "predicted_max_depth_cm": float(predicted_water_result["max_depth_cm"]),
        "gt_max_depth_cm": true_max_cm,
        "max_depth_error_cm": float(predicted_water_result["max_depth_cm"] - true_max_cm),
    }


def shoreline_membership_analysis(
    ray_records: list[dict[str, Any]],
    true_camera_mask: np.ndarray,
    true_outer_boundary: np.ndarray,
    true_water_level_m: float,
) -> dict[str, Any]:
    target_points = np.column_stack(np.where(true_outer_boundary)).astype(np.float64)
    groups: dict[str, list[float]] = {"inside_gt_water": [], "outside_gt_water": []}
    distances: list[float] = []
    for record in ray_records:
        if record.get("hit_status") != "success":
            continue
        row = int(np.clip(round(float(record["pixel_v"])), 0, true_camera_mask.shape[0] - 1))
        col = int(np.clip(round(float(record["pixel_u"])), 0, true_camera_mask.shape[1] - 1))
        group = "inside_gt_water" if true_camera_mask[row, col] else "outside_gt_water"
        groups[group].append(float(record["dem_z_m"]))
        if target_points.size:
            delta = target_points - np.asarray([row, col], dtype=np.float64)
            distances.append(float(np.sqrt(np.min(np.sum(delta * delta, axis=1)))))

    def stats(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean_height_m": float(np.mean(array)) if array.size else None,
            "median_height_m": float(np.median(array)) if array.size else None,
            "mad_m": float(np.median(np.abs(array - np.median(array)))) if array.size else None,
            "mean_signed_height_error_from_gt_level_m": float(np.mean(array - true_water_level_m)) if array.size else None,
        }

    distance_array = np.asarray(distances, dtype=np.float64)
    return {
        "classification_basis": "sampled predicted shoreline pixel membership in strict Camera GT",
        "inside_gt_water": stats(groups["inside_gt_water"]),
        "outside_gt_water": stats(groups["outside_gt_water"]),
        "within_2px_of_gt_outer_boundary_count": int(np.count_nonzero(distance_array <= 2.0)),
        "within_2px_of_gt_outer_boundary_ratio": float(np.mean(distance_array <= 2.0)) if distance_array.size else 0.0,
        "within_5px_of_gt_outer_boundary_count": int(np.count_nonzero(distance_array <= 5.0)),
        "within_5px_of_gt_outer_boundary_ratio": float(np.mean(distance_array <= 5.0)) if distance_array.size else 0.0,
    }


def gt_shoreline_geometry_counterfactual(
    true_camera_mask: np.ndarray,
    ground_dem: np.ndarray,
    sensors: dict[str, Any],
    mapping_config: dict[str, Any],
    true_water_level_m: float,
) -> dict[str, Any]:
    """Evaluation-only counterfactual; never replaces the saved prediction."""
    intersections, ray = intersect_camera_shoreline(
        np.where(true_camera_mask, 255, 0).astype(np.uint8), ground_dem, sensors, mapping_config
    )
    level, diagnostics = estimate_water_level_from_shoreline(
        intersections, mapping_config["shoreline_water_level"]
    )
    return {
        "data_role": "evaluation_only_gt_shoreline_counterfactual",
        "may_modify_prediction": False,
        "intersection_count": len(intersections),
        "intersection_success_rate": ray["shoreline_intersection_success_rate"],
        "estimated_from_gt_camera_shoreline_m": float(level),
        "gt_water_level_m": float(true_water_level_m),
        "signed_error_m": float(level - true_water_level_m),
        "absolute_error_m": abs(float(level - true_water_level_m)),
        "shoreline_height_mad_m": diagnostics["shoreline_height_mad_m"],
        "shoreline_height_iqr_m": diagnostics["shoreline_height_iqr_m"],
    }


def error_source_analysis(
    camera: dict[str, Any],
    boundary: dict[str, Any],
    level: dict[str, Any],
    dem_mask: dict[str, Any],
    reconstruction_at_predicted_level: dict[str, Any],
    ray_residual_p95_m: float,
    gt_shoreline_counterfactual_result: dict[str, Any],
    prediction_gate: dict[str, Any],
) -> dict[str, Any]:
    signals = {
        "segmentation_scope": bool(camera["iou"] < 0.80 or camera["area_pixel_relative_error"] > 0.25),
        "shoreline_localization": bool(boundary["symmetric_outer_boundary"]["p95_px"] > 5.0),
        "camera_geometry": bool(
            ray_residual_p95_m > 1e-4
            or gt_shoreline_counterfactual_result["absolute_error_m"] > 0.02
        ),
        "water_level_estimation": bool(level["absolute_water_level_error_m"] > 0.02),
        "dem_reconstruction": bool(reconstruction_at_predicted_level["iou"] < 0.95),
    }
    if signals["segmentation_scope"] and signals["shoreline_localization"]:
        dominant = "mixed"
    elif signals["segmentation_scope"]:
        dominant = "segmentation_scope"
    elif signals["shoreline_localization"]:
        dominant = "shoreline_localization"
    elif signals["camera_geometry"]:
        dominant = "camera_geometry"
    elif signals["water_level_estimation"]:
        dominant = "water_level_estimation"
    elif signals["dem_reconstruction"]:
        dominant = "dem_reconstruction"
    else:
        dominant = "mixed"
    if camera["recall"] >= 0.90 and camera["precision"] >= 0.90:
        scope_interpretation = "complete_water_region_like"
    elif camera["precision"] < 0.80 and camera["predicted_pixels"] > camera["gt_pixels"]:
        scope_interpretation = "overexpanded_ripple_or_reflection_region_not_complete_physical_water"
    else:
        scope_interpretation = "partial_ripple_or_reflection_region_not_complete_physical_water"
    return {
        "dominant_error_source": dominant,
        "error_signals": signals,
        "prediction_side_reject_status": prediction_gate.get("status"),
        "prediction_side_reject_reasons": prediction_gate.get("reasons", []),
        "prediction_side_reject_consistent_with_gt_evaluation": bool(
            prediction_gate.get("status") == "reject"
            and (camera["iou"] < 0.90 or boundary["symmetric_outer_boundary"]["p95_px"] > 3.0)
        ),
        "manual_prompt_scope_interpretation": scope_interpretation,
        "reasoning": {
            "camera_mask_scope": "Camera mask size and overlap quantify over/under-coverage before geometry.",
            "shoreline": "Outer-boundary distance excludes enclosed holes.",
            "geometry": "Ray residual and GT-shoreline counterfactual separate numerical geometry from candidate shoreline error.",
            "water_level": "Predicted shoreline heights explain the saved robust-median level; no answer data changes it.",
            "reconstruction": "Saved DEM mask is compared with the below-predicted-level lowland only as an evaluation decomposition.",
        },
        "dem_mask_iou": dem_mask["iou"],
    }
