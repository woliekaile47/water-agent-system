"""Tests for Phase 2D-C-6B-4 frozen failure-mode audit."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.evaluation.analyze_temporal_sam2_prompt_failures import (
    classify_failure_mode,
    prompt_ground_truth_support,
)
from src.vision import temporal_sam2_prompt_pipeline as prediction


def test_prompt_gt_support_measures_points_box_and_error_location_without_mutation() -> None:
    truth = np.zeros((10, 12), dtype=bool)
    truth[3:7, 3:9] = True
    predicted = truth.copy()
    predicted[3:7, 9] = True
    before = predicted.copy()
    prompt = {
        "box_xyxy": [2, 2, 9, 8],
        "positive_points_xy": [[4, 4], [0, 0]],
        "negative_points_xy": [[1, 1], [5, 5]],
    }
    result = prompt_ground_truth_support(prompt, predicted, truth)
    assert result["positive_points_in_gt_water"] == 1
    assert result["positive_point_gt_support_rate"] == 0.5
    assert result["negative_points_outside_gt_water"] == 1
    assert result["negative_point_gt_correct_rate"] == 0.5
    assert result["box_gt_water_coverage"] == 1.0
    assert result["false_positive_outside_box_fraction"] == 0.0
    assert np.array_equal(predicted, before)


def test_prompt_gt_support_rejects_out_of_range_points() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    prompt = {"box_xyxy": [0, 0, 4, 4], "positive_points_xy": [[5, 1]], "negative_points_xy": []}
    with pytest.raises(ValueError, match="outside"):
        prompt_ground_truth_support(prompt, mask, mask)


def test_failure_classification_distinguishes_prompt_contamination_and_scope_truncation() -> None:
    over = classify_failure_mode(
        {"iou": 0.48, "precision": 0.48, "recall": 0.99},
        28.0,
        {"positive_point_gt_support_rate": 0.4, "box_gt_water_coverage": 1.0, "false_negative_outside_box_fraction": 0.0},
    )
    assert over["evaluation_failure_mode"] == "oversegmentation"
    assert over["diagnostic_attribution"] == "positive_prompt_contamination"
    under = classify_failure_mode(
        {"iou": 0.72, "precision": 0.999, "recall": 0.72},
        69.0,
        {"positive_point_gt_support_rate": 1.0, "box_gt_water_coverage": 0.73, "false_negative_outside_box_fraction": 0.91},
    )
    assert under["evaluation_failure_mode"] == "undersegmentation"
    assert under["diagnostic_attribution"] == "prompt_box_scope_truncation"


def test_accurate_mask_is_not_reclassified_as_failure() -> None:
    result = classify_failure_mode(
        {"iou": 0.95, "precision": 0.96, "recall": 0.97},
        4.0,
        {"positive_point_gt_support_rate": 1.0, "box_gt_water_coverage": 1.0, "false_negative_outside_box_fraction": 0.0},
    )
    assert result["evaluation_failure_mode"] == "none_observed"
    assert result["offline_research_criteria_met"] is True
    assert result["not_a_prediction_side_gate"] is True


def test_prediction_code_has_no_failure_audit_or_gt_dependency() -> None:
    source = inspect.getsource(prediction)
    for token in ("prompt_ground_truth_support", "positive_point_gt_support_rate", "camera_water_mask_gt"):
        assert token not in source
