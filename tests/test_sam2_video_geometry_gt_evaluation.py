"""Tests for independent C7-4 scalar geometry GT evaluation."""

from __future__ import annotations

import inspect

import numpy as np

from src.evaluation import evaluate_sam2_video_geometry_gt as evaluation


def test_depth_scalar_statistics_use_requested_domain() -> None:
    depth = np.asarray([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32)
    mask = np.asarray([[False, True], [True, False]])
    result = evaluation.depth_scalar_statistics(depth, mask)
    assert result["cell_count"] == 2
    assert np.isclose(result["mean_depth_cm"], 15.0)
    assert np.isclose(result["median_depth_cm"], 15.0)
    assert np.isclose(result["max_depth_cm"], 20.0)


def test_scalar_frame_evaluation_distinguishes_visible_and_global_area() -> None:
    prediction = {
        "frame_index": 99,
        "geometry_available": True,
        "quality_status": "reject",
        "gate_reasons": ["candidate_basin_outside_camera_coverage"],
        "global_estimate_status": "partial",
        "observable_region_result_valid": True,
        "estimated_water_level_m": 0.11,
        "water_area_m2": 8.0,
        "water_volume_m3": 0.8,
        "mean_depth_cm": 10.0,
        "median_depth_cm": 9.0,
        "max_depth_cm": 20.0,
    }
    truth = {"water_level_m": 0.1}
    basin = {
        "camera_visible_main_basin": {
            "area_m2": 8.0,
            "volume_m3": 0.8,
            "depth": {"mean_depth_cm": 10.0, "median_depth_cm": 9.0, "max_depth_cm": 20.0},
        },
        "global_scene": {
            "area_m2": 10.0,
            "volume_m3": 1.0,
            "depth": {"mean_depth_cm": 10.0, "median_depth_cm": 9.0, "max_depth_cm": 20.0},
        },
    }
    result = evaluation.evaluate_frozen_scalar_frame(prediction, truth, basin)
    assert result["water_level"]["within_project_3cm_target"] is True
    assert result["area_camera_visible_main_basin_m2"]["relative_error"] == 0.0
    assert result["area_global_scene_m2"]["relative_error"] == 0.2
    assert result["per_cell_dem_mask_metrics"] is None


def test_summary_reports_rejected_but_accurate_frames() -> None:
    rows = []
    for status, error_cm in (("pass", 0.2), ("reject", 0.5), ("reject", 3.5)):
        rows.append({
            "evaluation_available": True,
            "prediction_side_quality_status": status,
            "water_level": {"absolute_error_cm": error_cm, "within_project_3cm_target": error_cm <= 3.0},
            "area_camera_visible_main_basin_m2": {"relative_error": 0.1},
            "area_global_scene_m2": {"relative_error": 0.2},
            "volume_camera_visible_main_basin_m3": {"relative_error": 0.1},
            "volume_global_scene_m3": {"relative_error": 0.2},
            "mean_depth_camera_visible_main_basin_cm": {"absolute_error": 0.3},
            "max_depth_global_scene_cm": {"absolute_error": 0.4},
        })
    summary = evaluation.summarize_scalar_evaluation(rows)
    assert summary["water_level_within_3cm_count"] == 2
    assert summary["quality_status_vs_3cm_target"]["reject"]["within_3cm_count"] == 1


def test_visible_basin_is_selected_by_camera_overlap_without_merging_unobservable(
    monkeypatch,
) -> None:
    visible = np.asarray([[True, True, False, False], [False, False, False, False]])
    hidden = np.asarray([[False, False, False, False], [False, False, True, True]])

    monkeypatch.setattr(evaluation, "connected_components", lambda _mask, _connectivity: [visible, hidden])

    def fake_reproject(component, _water_level, _sensors):
        projected = np.zeros((2, 4), dtype=np.uint8)
        if np.array_equal(component, visible):
            projected[0, :2] = 255
        return projected, {"water_surface_projection_coverage": float(np.any(projected))}

    monkeypatch.setattr(evaluation, "reproject_water_surface", fake_reproject)
    ground_truth = {
        "dem_mask": visible | hidden,
        "depth_map": np.asarray([[0.1, 0.1, 0.0, 0.0], [0.0, 0.0, 0.01, 0.01]]),
        "camera_mask": visible.copy(),
        "water_level_m": 0.2,
        "water_area_m2": 0.04,
        "water_volume_m3": 0.0022,
    }
    result = evaluation.derive_camera_visible_basin_ground_truth(
        ground_truth, {"road": {"dem_resolution_m": 0.1}}
    )

    assert result["component_count"] == 2
    assert result["camera_visible_main_component_index"] == 0
    assert result["unobservable_component_count"] == 1
    assert result["camera_visible_main_basin"]["cell_count"] == 2
    assert result["global_scene"]["area_m2"] == 0.04


def test_evaluation_module_is_separate_from_prediction() -> None:
    source = inspect.getsource(evaluation)
    assert "run_sam2_video_geometry_stability" not in source
    assert "run_video_frame_geometry" not in source
    assert "build_sam2" not in source
    assert "prediction_recomputed" in source
