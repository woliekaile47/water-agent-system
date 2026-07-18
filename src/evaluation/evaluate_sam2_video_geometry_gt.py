#!/usr/bin/env python3
"""Independent scalar GT evaluation for frozen C7 video geometry results."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.fusion.water_surface_aware_mask_to_dem import reproject_water_surface
from src.hydrology.estimate_water_level_from_boundary import connected_components


def depth_scalar_statistics(depth_map: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    depth = np.asarray(depth_map, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool) & np.isfinite(depth)
    values = depth[selected]
    if values.size == 0:
        return {
            "cell_count": 0,
            "mean_depth_cm": None,
            "median_depth_cm": None,
            "max_depth_cm": None,
        }
    return {
        "cell_count": int(values.size),
        "mean_depth_cm": float(np.mean(values) * 100.0),
        "median_depth_cm": float(np.median(values) * 100.0),
        "max_depth_cm": float(np.max(values) * 100.0),
    }


def derive_camera_visible_basin_ground_truth(
    ground_truth: dict[str, Any],
    sensors: dict[str, Any],
) -> dict[str, Any]:
    """Describe GT components for evaluation without changing prediction selection."""
    dem_mask = np.asarray(ground_truth["dem_mask"], dtype=bool)
    depth = np.asarray(ground_truth["depth_map"], dtype=np.float64)
    camera_mask = np.asarray(ground_truth["camera_mask"], dtype=bool)
    cell_size = float(sensors["road"]["dem_resolution_m"])
    cell_area = cell_size * cell_size
    components = connected_components(dem_mask, 8)
    records: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        projected, projection = reproject_water_surface(
            component, float(ground_truth["water_level_m"]), sensors
        )
        projected_mask = projected > 127
        records.append({
            "component_index": index,
            "cell_count": int(np.count_nonzero(component)),
            "area_m2": float(np.count_nonzero(component) * cell_area),
            "volume_m3": float(np.sum(depth[component]) * cell_area),
            "camera_projected_pixels": int(np.count_nonzero(projected_mask)),
            "camera_gt_overlap_pixels": int(np.count_nonzero(projected_mask & camera_mask)),
            "projection_coverage": float(projection.get("water_surface_projection_coverage", 0.0)),
            "depth": depth_scalar_statistics(depth, component),
        })
    if not records:
        raise ValueError("GT contains no water component")
    selected = max(
        records,
        key=lambda item: (
            item["camera_gt_overlap_pixels"],
            item["camera_projected_pixels"],
            item["cell_count"],
        ),
    )
    selected_index = int(selected["component_index"])
    for record in records:
        record["camera_visible_main_basin"] = record["component_index"] == selected_index
        record["camera_observable"] = bool(
            record["camera_projected_pixels"] > 0 and record["camera_gt_overlap_pixels"] > 0
        )
    return {
        "selection_role": "independent_evaluation_only_not_prediction_input",
        "selection_method": "maximum_camera_gt_projection_overlap_then_projection_then_cell_count",
        "component_count": len(records),
        "camera_visible_main_component_index": selected_index,
        "unobservable_component_count": sum(not record["camera_observable"] for record in records),
        "components": records,
        "camera_visible_main_basin": selected,
        "global_scene": {
            "area_m2": float(ground_truth["water_area_m2"]),
            "volume_m3": float(ground_truth["water_volume_m3"]),
            "depth": depth_scalar_statistics(depth, dem_mask),
        },
        "ground_truth_used_for_evaluation_only": True,
    }


def _error(predicted: float, truth: float) -> dict[str, float | None]:
    signed = float(predicted - truth)
    absolute = abs(signed)
    return {
        "predicted": float(predicted),
        "ground_truth": float(truth),
        "signed_error": signed,
        "absolute_error": absolute,
        "relative_error": None if truth == 0.0 else float(absolute / abs(truth)),
    }


def evaluate_frozen_scalar_frame(
    prediction_row: dict[str, Any],
    ground_truth: dict[str, Any],
    basin_truth: dict[str, Any],
) -> dict[str, Any]:
    if not prediction_row.get("geometry_available"):
        return {
            "frame_index": int(prediction_row["frame_index"]),
            "evaluation_available": False,
            "failure_reason": "frozen_prediction_geometry_unavailable",
            "ground_truth_used_for_prediction": False,
        }
    visible = basin_truth["camera_visible_main_basin"]
    global_scene = basin_truth["global_scene"]
    water_level = _error(
        float(prediction_row["estimated_water_level_m"]),
        float(ground_truth["water_level_m"]),
    )
    return {
        "frame_index": int(prediction_row["frame_index"]),
        "evaluation_available": True,
        "prediction_side_quality_status": prediction_row["quality_status"],
        "prediction_side_gate_reasons": list(prediction_row.get("gate_reasons", [])),
        "prediction_global_estimate_status": prediction_row.get("global_estimate_status"),
        "prediction_observable_region_result_valid": bool(
            prediction_row.get("observable_region_result_valid", False)
        ),
        "water_level": {
            **water_level,
            "signed_error_cm": float(water_level["signed_error"] * 100.0),
            "absolute_error_cm": float(water_level["absolute_error"] * 100.0),
            "within_project_3cm_target": bool(water_level["absolute_error"] <= 0.03),
        },
        "area_global_scene_m2": _error(
            float(prediction_row["water_area_m2"]), float(global_scene["area_m2"])
        ),
        "area_camera_visible_main_basin_m2": _error(
            float(prediction_row["water_area_m2"]), float(visible["area_m2"])
        ),
        "volume_global_scene_m3": _error(
            float(prediction_row["water_volume_m3"]), float(global_scene["volume_m3"])
        ),
        "volume_camera_visible_main_basin_m3": _error(
            float(prediction_row["water_volume_m3"]), float(visible["volume_m3"])
        ),
        "mean_depth_global_scene_cm": _error(
            float(prediction_row["mean_depth_cm"]), float(global_scene["depth"]["mean_depth_cm"])
        ),
        "mean_depth_camera_visible_main_basin_cm": _error(
            float(prediction_row["mean_depth_cm"]), float(visible["depth"]["mean_depth_cm"])
        ),
        "median_depth_global_scene_cm": _error(
            float(prediction_row["median_depth_cm"]), float(global_scene["depth"]["median_depth_cm"])
        ),
        "median_depth_camera_visible_main_basin_cm": _error(
            float(prediction_row["median_depth_cm"]), float(visible["depth"]["median_depth_cm"])
        ),
        "max_depth_global_scene_cm": _error(
            float(prediction_row["max_depth_cm"]), float(global_scene["depth"]["max_depth_cm"])
        ),
        "per_cell_dem_mask_metrics": None,
        "per_cell_depth_mae_rmse_bias": None,
        "per_cell_metrics_unavailable_reason": (
            "C7-3 froze per-frame scalar geometry but only saved DEM/depth arrays for the anchor; "
            "prediction was not recomputed for evaluation."
        ),
        "ground_truth_used_for_evaluation_only": True,
        "ground_truth_used_for_prediction": False,
        "eligible_for_downstream": False,
    }


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None, "std": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def summarize_scalar_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row["evaluation_available"]]
    if not available:
        return {"frame_count": len(rows), "available_frame_count": 0}

    def collect(path: tuple[str, ...]) -> list[float]:
        values = []
        for row in available:
            value: Any = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return values

    statuses = sorted({row["prediction_side_quality_status"] for row in available})
    return {
        "frame_count": len(rows),
        "available_frame_count": len(available),
        "water_level_absolute_error_cm": metric_summary(collect(("water_level", "absolute_error_cm"))),
        "water_level_within_3cm_count": sum(
            bool(row["water_level"]["within_project_3cm_target"]) for row in available
        ),
        "visible_area_relative_error": metric_summary(
            collect(("area_camera_visible_main_basin_m2", "relative_error"))
        ),
        "global_area_relative_error": metric_summary(collect(("area_global_scene_m2", "relative_error"))),
        "visible_volume_relative_error": metric_summary(
            collect(("volume_camera_visible_main_basin_m3", "relative_error"))
        ),
        "global_volume_relative_error": metric_summary(
            collect(("volume_global_scene_m3", "relative_error"))
        ),
        "visible_mean_depth_absolute_error_cm": metric_summary(
            collect(("mean_depth_camera_visible_main_basin_cm", "absolute_error"))
        ),
        "global_max_depth_absolute_error_cm": metric_summary(
            collect(("max_depth_global_scene_cm", "absolute_error"))
        ),
        "quality_status_vs_3cm_target": {
            status: {
                "frame_count": sum(row["prediction_side_quality_status"] == status for row in available),
                "within_3cm_count": sum(
                    row["prediction_side_quality_status"] == status
                    and row["water_level"]["within_project_3cm_target"]
                    for row in available
                ),
            }
            for status in statuses
        },
        "per_cell_metrics_available": False,
        "prediction_recomputed": False,
        "eligible_for_downstream": False,
    }
