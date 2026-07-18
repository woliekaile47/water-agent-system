#!/usr/bin/env python3
"""Evaluation-only diagnostics for frozen temporal prompts and SAM 2 masks."""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / max(1, denominator))


def prompt_ground_truth_support(
    prompt: dict[str, Any],
    predicted_mask: np.ndarray,
    camera_gt_mask: np.ndarray,
) -> dict[str, Any]:
    """Measure frozen prompt support against GT without modifying prediction."""
    prediction = np.asarray(predicted_mask, dtype=bool)
    truth = np.asarray(camera_gt_mask, dtype=bool)
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and Camera GT shapes differ")
    height, width = truth.shape
    positive = [[int(v) for v in point] for point in prompt["positive_points_xy"]]
    negative = [[int(v) for v in point] for point in prompt["negative_points_xy"]]
    for x, y in positive + negative:
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Prompt point is outside the Camera image")
    x1, y1, x2, y2 = [int(v) for v in prompt["box_xyxy"]]
    if x1 > x2 or y1 > y2:
        raise ValueError("Prompt box coordinates are reversed")
    box = np.zeros_like(truth)
    box[max(0, y1) : min(height, y2 + 1), max(0, x1) : min(width, x2 + 1)] = True
    if not np.any(box):
        raise ValueError("Prompt box does not overlap the Camera image")

    positive_in_water = sum(bool(truth[y, x]) for x, y in positive)
    negative_outside_water = sum(not bool(truth[y, x]) for x, y in negative)
    false_positive = prediction & ~truth
    false_negative = ~prediction & truth
    gt_pixels = int(np.count_nonzero(truth))
    predicted_pixels = int(np.count_nonzero(prediction))
    box_pixels = int(np.count_nonzero(box))
    gt_in_box = int(np.count_nonzero(truth & box))
    fp_count = int(np.count_nonzero(false_positive))
    fn_count = int(np.count_nonzero(false_negative))
    return {
        "positive_point_count": len(positive),
        "positive_points_in_gt_water": positive_in_water,
        "positive_point_gt_support_rate": _safe_fraction(positive_in_water, len(positive)),
        "negative_point_count": len(negative),
        "negative_points_outside_gt_water": negative_outside_water,
        "negative_point_gt_correct_rate": _safe_fraction(negative_outside_water, len(negative)),
        "box_xyxy": [x1, y1, x2, y2],
        "box_pixels": box_pixels,
        "box_gt_water_coverage": _safe_fraction(gt_in_box, gt_pixels),
        "box_water_purity": _safe_fraction(gt_in_box, box_pixels),
        "predicted_to_gt_area_ratio": _safe_fraction(predicted_pixels, gt_pixels),
        "false_positive_outside_box_fraction": _safe_fraction(
            int(np.count_nonzero(false_positive & ~box)), fp_count
        ),
        "false_negative_outside_box_fraction": _safe_fraction(
            int(np.count_nonzero(false_negative & ~box)), fn_count
        ),
        "prediction_modified": False,
        "ground_truth_role": "independent_evaluation_only",
    }


def classify_failure_mode(
    camera_metrics: dict[str, Any],
    outer_boundary_p95_px: float | None,
    support: dict[str, Any],
) -> dict[str, Any]:
    """Classify an observed evaluation failure; this is not a gate decision."""
    precision = float(camera_metrics["precision"])
    recall = float(camera_metrics["recall"])
    iou = float(camera_metrics["iou"])
    boundary_ok = outer_boundary_p95_px is not None and float(outer_boundary_p95_px) <= 5.0
    all_met = iou >= 0.90 and recall >= 0.90 and boundary_ok
    if all_met:
        mode = "none_observed"
        attribution = "offline_research_criteria_met"
    elif precision < 0.90 and recall >= 0.90:
        mode = "oversegmentation"
        if float(support["positive_point_gt_support_rate"]) <= 0.60:
            attribution = "positive_prompt_contamination"
        else:
            attribution = "sam2_scope_expansion_or_boundary_ambiguity"
    elif recall < 0.90 and precision >= 0.90:
        mode = "undersegmentation"
        if (
            float(support["box_gt_water_coverage"]) < 0.90
            and float(support["false_negative_outside_box_fraction"]) > 0.50
        ):
            attribution = "prompt_box_scope_truncation"
        else:
            attribution = "sam2_incomplete_scope_or_weak_visual_evidence"
    elif precision < 0.90 and recall < 0.90:
        mode = "mixed_segmentation_error"
        attribution = "mixed_prompt_and_segmentation_scope"
    else:
        mode = "boundary_or_iou_only"
        attribution = "localized_boundary_mismatch"
    return {
        "evaluation_failure_mode": mode,
        "diagnostic_attribution": attribution,
        "offline_research_criteria_met": all_met,
        "not_a_prediction_side_gate": True,
        "authoritative": False,
    }
