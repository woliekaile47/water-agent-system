#!/usr/bin/env python3
"""Independent post-prediction evaluation for Phase 2D-A outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from src.evaluation.evaluate_simulation_depth import load_ground_truth_evaluation_inputs
from src.evaluation.evaluate_temporal_water_mask import evaluate_water_mask
from src.fusion.water_surface_aware_mask_to_dem import reproject_water_surface
from src.hydrology.estimate_water_level_from_boundary import connected_components, dilate_mask
from src.hydrology.invert_boundary_waterline_depth import extract_boundary_mask


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _completed_prediction(output: Path) -> dict[str, Any]:
    required = {
        "manifest": output / "prediction_manifest.json",
        "visual_gate": output / "visual_quality_gate.json",
        "geometry_gate": output / "geometry_quality_gate.json",
        "integration_gate": output / "integration_quality_gate.json",
        "camera_mask": output / "predicted_camera_water_mask.png",
        "unknown_mask": output / "predicted_camera_unknown_mask.png",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prediction is incomplete; evaluation cannot read GT: {missing}")
    documents = {name: _read_json(path) for name, path in required.items() if name not in ("camera_mask", "unknown_mask")}
    if documents["manifest"].get("data_role") != "prediction":
        raise ValueError("prediction_manifest data_role is not prediction")
    if documents["manifest"].get("ground_truth_or_metadata_read_during_prediction") is not False:
        raise ValueError("prediction manifest does not prove GT isolation")
    documents.update({"paths": required})
    return documents


def _camera_metrics(predicted: np.ndarray, unknown: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    metrics = evaluate_water_mask(predicted, unknown, truth)
    return {
        "whole_image_iou": metrics["whole_image_iou"],
        "known_region_iou": metrics["evaluated_known_region_iou"],
        "precision": metrics["pixel_precision"], "recall": metrics["pixel_recall"],
        "f1": metrics["pixel_f1"], "unknown_fraction": metrics["unknown_fraction"],
        "predicted_water_pixels": metrics["predicted_water_area_pixels"],
        "gt_water_pixels": metrics["gt_water_area_pixels"],
    }


def _mask_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    pred, gt = np.asarray(predicted, dtype=bool), np.asarray(truth, dtype=bool)
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    pred_boundary, gt_boundary = extract_boundary_mask(pred), extract_boundary_mask(gt)
    boundary_precision = _safe_ratio(
        np.count_nonzero(pred_boundary & dilate_mask(gt_boundary, 1)), np.count_nonzero(pred_boundary),
    )
    boundary_recall = _safe_ratio(
        np.count_nonzero(gt_boundary & dilate_mask(pred_boundary, 1)), np.count_nonzero(gt_boundary),
    )
    boundary_f1 = None
    if boundary_precision is not None and boundary_recall is not None and boundary_precision + boundary_recall:
        boundary_f1 = float(2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall))
    return {
        "iou": _safe_ratio(tp, np.count_nonzero(pred | gt)),
        "precision": _safe_ratio(tp, tp + fp), "recall": _safe_ratio(tp, tp + fn),
        "boundary_f1_1cell": boundary_f1,
        "intersection_cells": tp, "false_positive_cells": fp, "false_negative_cells": fn,
    }


def _depth_domain(prediction: np.ndarray, truth: np.ndarray, domain: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(domain, dtype=bool) & np.isfinite(prediction) & np.isfinite(truth)
    if not np.any(valid):
        return {"cell_count": 0, "mae_m": None, "rmse_m": None}
    error = prediction[valid].astype(np.float64) - truth[valid].astype(np.float64)
    return {"cell_count": int(error.size), "mae_m": float(np.mean(np.abs(error))),
            "rmse_m": float(np.sqrt(np.mean(error * error)))}


def _load_sensors(project_root: Path) -> dict[str, Any]:
    with (project_root / "simulation/config/sensors.yaml").open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _component_evaluation_40cm(
    gt_mask: np.ndarray, predicted_mask: np.ndarray | None, water_level: float,
    camera_unknown: np.ndarray, project_root: Path,
) -> list[dict[str, Any]]:
    sensors = _load_sensors(project_root)
    results = []
    components = connected_components(np.asarray(gt_mask, dtype=bool), 8)
    sizes = [int(np.count_nonzero(component)) for component in components]
    main_index = int(np.argmax(sizes)) if sizes else -1
    for index, component in enumerate(components):
        projected, _ = reproject_water_surface(component, water_level, sensors)
        projected_mask = projected > 127
        projected_pixels = int(np.count_nonzero(projected_mask))
        known_pixels = int(np.count_nonzero(projected_mask & ~camera_unknown))
        intersection = int(np.count_nonzero(component & predicted_mask)) if predicted_mask is not None else 0
        observable = bool(projected_pixels > 0 and known_pixels > 0)
        results.append({
            "component_index": index, "ground_truth_cell_count": sizes[index],
            "prediction_intersection_cell_count": intersection,
            "recall": float(intersection / max(1, sizes[index])) if predicted_mask is not None else None,
            "camera_projected_pixels": projected_pixels, "camera_known_projected_pixels": known_pixels,
            "camera_observable_or_not": "observable" if observable else "unobservable",
            "component_role": "visible_main_basin" if index == main_index and observable else "unobservable_secondary_basin" if not observable else "observable_secondary_basin",
            "unobservable_false_negative_is_reported_separately": not observable,
        })
    return results


def evaluate_synthetic_visual_to_depth_case(
    sequence_dir: str | Path, prediction_output_dir: str | Path, project_root: str | Path,
) -> dict[str, Any]:
    sequence = Path(sequence_dir).expanduser().resolve()
    output = Path(prediction_output_dir).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    prediction = _completed_prediction(output)  # Must complete before either GT loader is called.
    case_id, rain_level, seed_name = sequence.parents[1].name, sequence.parent.name, sequence.name
    is_dry = case_id == "sim_dry_baseline_001"
    predicted_camera = np.asarray(Image.open(prediction["paths"]["camera_mask"]).convert("L"), dtype=np.uint8) > 127
    unknown_camera = np.asarray(Image.open(prediction["paths"]["unknown_mask"]).convert("L"), dtype=np.uint8) > 127
    dynamic_water_gt = np.asarray(
        Image.open(sequence / "ground_truth/water_mask.png").convert("L"), dtype=np.uint8,
    ) > 127
    camera_metrics = _camera_metrics(predicted_camera, unknown_camera, dynamic_water_gt)
    visual_gate = prediction["visual_gate"]
    geometry_gate = prediction["geometry_gate"]
    integration_gate = prediction["integration_gate"]
    gate = {
        "visual_gate_status": visual_gate.get("status"),
        "geometry_gate_status": geometry_gate.get("status"),
        "integration_gate_status": integration_gate.get("status"),
        "measurement_status": integration_gate.get("measurement_status"),
        "authoritative_measurement_available": bool(integration_gate.get("authoritative_measurement_available", False)),
        "candidate_values_available": bool(integration_gate.get("candidate_values_available", False)),
        "global_estimate_status": integration_gate.get("global_estimate_status"),
        "observable_region_result_valid": bool(integration_gate.get("observable_region_result_valid", False)),
        "area_volume_semantics": integration_gate.get("area_volume_semantics"),
        "reject_reasons": integration_gate.get("reasons", []),
    }
    base = {
        "data_role": "evaluation", "ground_truth_used_for_evaluation_only": True,
        "case_id": case_id, "rain_level": rain_level, "seed_name": seed_name,
        "seed": int(seed_name.removeprefix("seed_")), "is_dry": is_dry,
        "nominal_depth_cm": None if is_dry else int(case_id.split("_")[-2].removesuffix("cm")),
        "camera_mask": camera_metrics, "gate": gate,
        "metric_role": "dry_false_positive_diagnostic" if is_dry else (
            "authoritative_measurement" if gate["authoritative_measurement_available"]
            else "rejected_candidate_diagnostic" if gate["candidate_values_available"]
            else "unavailable"
        ),
        "eligible_for_downstream": False,
    }
    if is_dry:
        component_count = cv2.connectedComponents(predicted_camera.astype(np.uint8), connectivity=8)[0] - 1
        false_area = None
        if gate["candidate_values_available"] and (output / "predicted_water_result.json").is_file():
            false_area = float(_read_json(output / "predicted_water_result.json")["water_area_m2"])
        elif not np.any(predicted_camera):
            false_area = 0.0
        base["dry_false_positive"] = {
            "false_water_pixels": int(np.count_nonzero(predicted_camera)),
            "false_water_fraction": float(np.mean(predicted_camera)),
            "false_water_components": int(component_count),
            "false_positive_area_m2": false_area,
        }
        base.update({"water_level": None, "dem_mask": None, "depth": None,
                     "area": None, "volume": None, "components_40cm": None})
        return base

    static_gt = load_ground_truth_evaluation_inputs(root, case_id)
    result_path = output / "predicted_water_result.json"
    mask_path = output / "predicted_dem_mask.npy"
    depth_path = output / "predicted_depth_map_m.npy"
    candidate_complete = gate["candidate_values_available"] and result_path.is_file() and mask_path.is_file() and depth_path.is_file()
    if not candidate_complete:
        base.update({"water_level": None, "dem_mask": None, "depth": None,
                     "area": None, "volume": None, "components_40cm": None})
        return base
    result = _read_json(result_path)
    predicted_mask = np.load(mask_path).astype(bool)
    predicted_depth = np.load(depth_path).astype(np.float32)
    gt_mask = np.asarray(static_gt["dem_mask"], dtype=bool)
    gt_depth = np.asarray(static_gt["depth_map"], dtype=np.float32)
    union = predicted_mask | gt_mask
    pred_values = predicted_depth[predicted_mask & np.isfinite(predicted_depth)]
    gt_values = gt_depth[gt_mask & np.isfinite(gt_depth)]
    predicted_level = float(result["predicted_water_level_m"])
    gt_level = float(static_gt["water_level_m"])
    predicted_area, gt_area = float(result["water_area_m2"]), float(static_gt["water_area_m2"])
    predicted_volume, gt_volume = float(result["water_volume_m3"]), float(static_gt["water_volume_m3"])
    base["water_level"] = {"predicted_water_level_m": predicted_level,
        "ground_truth_water_level_m": gt_level, "water_level_absolute_error_m": abs(predicted_level - gt_level)}
    base["dem_mask"] = _mask_metrics(predicted_mask, gt_mask)
    base["depth"] = {
        "ground_truth_water_region": _depth_domain(predicted_depth, gt_depth, gt_mask),
        "prediction_gt_union": _depth_domain(predicted_depth, gt_depth, union),
        "max_depth_absolute_error_m": abs((float(np.max(pred_values)) if pred_values.size else 0.0) - (float(np.max(gt_values)) if gt_values.size else 0.0)),
        "mean_depth_absolute_error_m": abs((float(np.mean(pred_values)) if pred_values.size else 0.0) - (float(np.mean(gt_values)) if gt_values.size else 0.0)),
    }
    base["area"] = {"predicted_m2": predicted_area, "ground_truth_m2": gt_area,
        "absolute_error_m2": abs(predicted_area - gt_area),
        "relative_error": _safe_ratio(abs(predicted_area - gt_area), gt_area)}
    base["volume"] = {"predicted_m3": predicted_volume, "ground_truth_m3": gt_volume,
        "absolute_error_m3": abs(predicted_volume - gt_volume),
        "relative_error": _safe_ratio(abs(predicted_volume - gt_volume), gt_volume)}
    base["components_40cm"] = _component_evaluation_40cm(
        gt_mask, predicted_mask, gt_level, unknown_camera, root,
    ) if case_id == "sim_water_40cm_001" else None
    return base
