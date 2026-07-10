#!/usr/bin/env python3
"""Ground Truth loader and evaluation for completed Phase 2A predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.hydrology.estimate_water_level_from_boundary import dilate_mask
from src.hydrology.invert_boundary_waterline_depth import extract_boundary_mask


def load_ground_truth_evaluation_inputs(project_root: str | Path, case_id: str) -> dict[str, Any]:
    """Load answer data. This function must only be called after prediction."""
    root = Path(project_root).expanduser().resolve()
    case_dir = root / "data" / "simulation" / case_id
    manifest_path = case_dir / "manifest.json"
    metadata_path = case_dir / "ground_truth" / "ground_truth_metadata.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    return {
        "water_level_m": float(manifest["water_level_m"]),
        "water_area_m2": float(manifest["water_area_m2"]),
        "water_volume_m3": float(manifest["water_volume_m3"]),
        "dem_mask": np.load(case_dir / "ground_truth" / "dem_water_mask_gt.npy").astype(bool),
        "depth_map": np.load(case_dir / "ground_truth" / "depth_map_gt_m.npy").astype(np.float32),
        "metadata": metadata,
        "paths": {
            "manifest": str(manifest_path),
            "dem_mask_gt": str(case_dir / "ground_truth" / "dem_water_mask_gt.npy"),
            "depth_map_gt": str(case_dir / "ground_truth" / "depth_map_gt_m.npy"),
            "ground_truth_metadata": str(metadata_path),
        },
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _depth_errors(prediction: np.ndarray, truth: np.ndarray, domain: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(domain, dtype=bool) & np.isfinite(prediction) & np.isfinite(truth)
    if not np.any(valid):
        return {"cell_count": 0, "mae_m": None, "rmse_m": None}
    error = prediction[valid].astype(np.float64) - truth[valid].astype(np.float64)
    return {
        "cell_count": int(error.size),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
    }


def evaluate_prediction(
    predicted_mask: np.ndarray,
    predicted_depth: np.ndarray,
    predicted_result: dict[str, Any],
    boundary_diagnostics: dict[str, Any],
    projection_diagnostics: dict[str, Any],
    ground_truth: dict[str, Any],
    quality_gate: dict[str, Any],
) -> dict[str, Any]:
    pred_mask = np.asarray(predicted_mask, dtype=bool)
    gt_mask = np.asarray(ground_truth["dem_mask"], dtype=bool)
    if pred_mask.shape != gt_mask.shape or predicted_depth.shape != ground_truth["depth_map"].shape:
        raise ValueError("Prediction and Ground Truth raster shapes differ")
    tp = int(np.count_nonzero(pred_mask & gt_mask))
    fp = int(np.count_nonzero(pred_mask & ~gt_mask))
    fn = int(np.count_nonzero(~pred_mask & gt_mask))
    union = pred_mask | gt_mask
    pred_boundary = extract_boundary_mask(pred_mask)
    gt_boundary = extract_boundary_mask(gt_mask)
    tolerance = 1
    boundary_precision = _safe_ratio(np.count_nonzero(pred_boundary & dilate_mask(gt_boundary, tolerance)), np.count_nonzero(pred_boundary))
    boundary_recall = _safe_ratio(np.count_nonzero(gt_boundary & dilate_mask(pred_boundary, tolerance)), np.count_nonzero(gt_boundary))
    boundary_f1 = None
    if boundary_precision is not None and boundary_recall is not None and boundary_precision + boundary_recall > 0:
        boundary_f1 = float(2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall))
    gt_depth = np.asarray(ground_truth["depth_map"], dtype=np.float32)
    predicted_level = float(predicted_result["predicted_water_level_m"])
    gt_level = float(ground_truth["water_level_m"])
    predicted_area = float(predicted_result["water_area_m2"])
    predicted_volume = float(predicted_result["water_volume_m3"])
    gt_area = float(ground_truth["water_area_m2"])
    gt_volume = float(ground_truth["water_volume_m3"])
    pred_values = predicted_depth[pred_mask & np.isfinite(predicted_depth)]
    gt_values = gt_depth[gt_mask & np.isfinite(gt_depth)]
    return {
        "data_role": "evaluation",
        "ground_truth_used_for_evaluation_only": True,
        "mask": {
            "intersection_cells": tp,
            "union_cells": int(np.count_nonzero(union)),
            "iou": _safe_ratio(tp, np.count_nonzero(union)),
            "precision": _safe_ratio(tp, tp + fp),
            "recall": _safe_ratio(tp, tp + fn),
            "boundary_precision_1cell": boundary_precision,
            "boundary_recall_1cell": boundary_recall,
            "boundary_f1_1cell": boundary_f1,
        },
        "water_level": {"predicted_m": predicted_level, "ground_truth_m": gt_level, "absolute_error_m": abs(predicted_level - gt_level)},
        "depth_summary_errors": {
            "max_depth_absolute_error_m": abs((float(np.max(pred_values)) if pred_values.size else 0.0) - (float(np.max(gt_values)) if gt_values.size else 0.0)),
            "mean_depth_absolute_error_m": abs((float(np.mean(pred_values)) if pred_values.size else 0.0) - (float(np.mean(gt_values)) if gt_values.size else 0.0)),
        },
        "depth_error_domains": {
            "gt_prediction_union": _depth_errors(predicted_depth, gt_depth, union),
            "ground_truth_water_region": _depth_errors(predicted_depth, gt_depth, gt_mask),
            "prediction_water_region": _depth_errors(predicted_depth, gt_depth, pred_mask),
        },
        "area": {
            "predicted_m2": predicted_area,
            "ground_truth_m2": gt_area,
            "absolute_error_m2": abs(predicted_area - gt_area),
            "relative_error": _safe_ratio(abs(predicted_area - gt_area), gt_area),
        },
        "volume": {
            "predicted_m3": predicted_volume,
            "ground_truth_m3": gt_volume,
            "absolute_error_m3": abs(predicted_volume - gt_volume),
            "relative_error": _safe_ratio(abs(predicted_volume - gt_volume), gt_volume),
        },
        "valid_boundary_sample_count": boundary_diagnostics.get("valid_boundary_sample_count"),
        "boundary_height_mad_m": boundary_diagnostics.get("boundary_height_mad_m"),
        "boundary_height_iqr_m": boundary_diagnostics.get("boundary_height_iqr_m"),
        "boundary_height_std_m": boundary_diagnostics.get("boundary_height_std_m"),
        "projection_coverage": projection_diagnostics.get("projection_coverage"),
        "quality_gate_status": quality_gate["status"],
        "eligible_for_downstream": False,
    }
