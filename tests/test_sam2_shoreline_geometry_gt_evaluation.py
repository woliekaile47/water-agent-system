"""Tests for independent Phase 2D-C-3D Ground Truth evaluation."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from src.evaluation import evaluate_sam2_shoreline_geometry_gt as evaluation
from src.fusion import sam2_shoreline_geometry_adapter as prediction_adapter


def test_binary_mask_metrics_are_exact_and_do_not_mutate_inputs() -> None:
    predicted = np.asarray([[1, 1], [0, 0]], dtype=bool)
    truth = np.asarray([[1, 0], [1, 0]], dtype=bool)
    predicted_before = predicted.copy()
    truth_before = truth.copy()

    metrics = evaluation.binary_mask_metrics(predicted, truth)

    assert metrics["predicted_pixels"] == 2
    assert metrics["gt_pixels"] == 2
    assert metrics["intersection_pixels"] == 1
    assert metrics["union_pixels"] == 3
    assert metrics["iou"] == 1.0 / 3.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["false_positive_pixels"] == 1
    assert metrics["false_negative_pixels"] == 1
    assert np.array_equal(predicted, predicted_before)
    assert np.array_equal(truth, truth_before)


def test_boundary_metrics_preserve_direction_and_exclude_holes_from_outer_boundary() -> None:
    shifted = np.zeros((12, 12), dtype=bool)
    target = np.zeros_like(shifted)
    shifted[3:8, 4:9] = True
    target[3:8, 3:8] = True
    directional = evaluation.boundary_metrics(shifted, target)
    assert directional["predicted_shoreline_to_gt_boundary"]["p95_px"] == 1.0
    assert directional["gt_boundary_to_predicted_shoreline"]["p95_px"] == 1.0

    with_hole = np.zeros((15, 15), dtype=bool)
    solid = np.zeros_like(with_hole)
    with_hole[3:12, 3:12] = True
    solid[3:12, 3:12] = True
    with_hole[6:9, 6:9] = False
    hole_metrics = evaluation.boundary_metrics(with_hole, solid)
    assert hole_metrics["symmetric_outer_boundary"]["p95_px"] == 0.0
    assert hole_metrics["symmetric_boundary"]["p95_px"] > 0.0


def test_depth_metrics_use_declared_evaluation_domains() -> None:
    predicted_depth = np.asarray([[0.2, 0.0], [0.4, 0.0]], dtype=np.float64)
    true_depth = np.asarray([[0.1, 0.3], [0.0, 0.0]], dtype=np.float64)
    predicted_mask = predicted_depth > 0.0
    true_mask = true_depth > 0.0

    metrics = evaluation.depth_metrics(predicted_depth, true_depth, predicted_mask, true_mask)

    assert metrics["full_valid_dem_domain"]["cell_count"] == 4
    assert np.isclose(metrics["full_valid_dem_domain"]["mae_m"], 0.2)
    assert np.isclose(metrics["full_valid_dem_domain"]["bias_m"], 0.05)
    assert metrics["predicted_region_domain"]["cell_count"] == 2
    assert np.isclose(metrics["predicted_region_domain"]["mae_m"], 0.25)
    assert metrics["gt_region_domain"]["cell_count"] == 2
    assert np.isclose(metrics["gt_region_domain"]["mae_m"], 0.2)
    assert metrics["overlap_region_domain"]["cell_count"] == 1
    assert np.isclose(metrics["overlap_region_domain"]["mae_m"], 0.1)


def test_area_volume_and_depth_summary_errors_are_exact() -> None:
    result = {
        "water_area_m2": 3.0,
        "water_volume_m3": 0.12,
        "mean_depth_cm": 4.0,
        "median_depth_cm": 3.0,
        "max_depth_cm": 8.0,
    }
    true_depth = np.asarray([[0.01, 0.03], [0.0, 0.05]], dtype=np.float64)
    true_mask = true_depth > 0.0

    metrics = evaluation.area_volume_metrics(result, 2.0, 0.08, true_depth, true_mask)

    assert metrics["area_absolute_error_m2"] == 1.0
    assert metrics["area_relative_error"] == 0.5
    assert np.isclose(metrics["volume_absolute_error_m3"], 0.04)
    assert np.isclose(metrics["volume_relative_error"], 0.5)
    assert np.isclose(metrics["gt_mean_depth_cm"], 3.0)
    assert np.isclose(metrics["mean_depth_error_cm"], 1.0)
    assert np.isclose(metrics["median_depth_error_cm"], 0.0)
    assert np.isclose(metrics["max_depth_error_cm"], 3.0)


def test_water_level_metrics_are_repeatable_and_do_not_define_datum_ratio() -> None:
    heights = np.asarray([-0.41, -0.40, -0.39, -0.36], dtype=np.float64)
    first = evaluation.water_level_metrics(-0.36, -0.39, heights, 0.05)
    second = evaluation.water_level_metrics(-0.36, -0.39, heights, 0.05)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert np.isclose(first["signed_water_level_error_m"], 0.03)
    assert np.isclose(first["absolute_water_level_error_cm"], 3.0)
    assert first["relative_water_level_error"] is None


def test_ground_truth_loader_is_confined_to_evaluation_module() -> None:
    evaluation_source = inspect.getsource(evaluation)
    prediction_source = inspect.getsource(prediction_adapter)
    prediction_runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_sam2_shoreline_geometry_diagnostic.py"
    ).read_text(encoding="utf-8")

    assert "load_ground_truth_evaluation_inputs" in evaluation_source
    assert "load_ground_truth_evaluation_inputs" not in prediction_source
    assert "load_ground_truth_evaluation_inputs" not in prediction_runner
    for answer_token in (
        "camera_water_mask_gt",
        "dem_water_mask_gt",
        "depth_map_gt_m",
        "water_level_gt",
    ):
        assert answer_token not in prediction_source
        assert answer_token not in prediction_runner


def test_metric_functions_are_deterministic_for_identical_arrays() -> None:
    predicted = np.zeros((10, 10), dtype=bool)
    truth = np.zeros_like(predicted)
    predicted[2:7, 3:8] = True
    truth[2:8, 2:7] = True
    first = {
        "mask": evaluation.binary_mask_metrics(predicted, truth),
        "boundary": evaluation.boundary_metrics(predicted, truth),
    }
    second = {
        "mask": evaluation.binary_mask_metrics(predicted, truth),
        "boundary": evaluation.boundary_metrics(predicted, truth),
    }
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
