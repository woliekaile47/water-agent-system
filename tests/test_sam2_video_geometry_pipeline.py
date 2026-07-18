"""Tests for Phase 2D-C-7-3 prediction-side video geometry."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.fusion import sam2_video_geometry_pipeline as pipeline


def test_main_outer_shoreline_excludes_fragment_and_ignores_hole_boundary() -> None:
    mask = np.zeros((80, 100), dtype=bool)
    mask[20:60, 30:80] = True
    mask[35:40, 50:55] = False
    mask[2, 2] = True
    before = mask.copy()
    first = pipeline.extract_main_outer_shoreline(mask)
    second = pipeline.extract_main_outer_shoreline(mask)
    assert first["component_count"] == 2
    assert first["excluded_fragment_pixels"] == 1
    assert first["sampled_shoreline_xy"].shape == (128, 2)
    assert first["full_shoreline_point_count"] == 176
    assert np.array_equal(first["sampled_shoreline_xy"], second["sampled_shoreline_xy"])
    assert np.array_equal(mask, before)
    assert not first["selected_mask"][2, 2]
    assert not first["selected_mask"][37, 52]


def test_outer_shoreline_coordinates_are_finite_and_in_bounds() -> None:
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:25, 8:32] = True
    result = pipeline.extract_main_outer_shoreline(mask)
    points = result["sampled_shoreline_xy"]
    assert np.isfinite(points).all()
    assert np.all((points[:, 0] >= 0) & (points[:, 0] < 40))
    assert np.all((points[:, 1] >= 0) & (points[:, 1] < 30))


def test_empty_mask_fails_closed() -> None:
    with pytest.raises(ValueError, match="empty"):
        pipeline.extract_main_outer_shoreline(np.zeros((20, 20), dtype=bool))


def test_video_result_semantics_are_not_authoritative() -> None:
    semantics = pipeline.video_prediction_semantics()
    assert semantics["ground_truth_used"] is False
    assert semantics["authoritative"] is False
    assert semantics["eligible_for_downstream"] is False
    assert semantics["eligible_for_formal_s5_s8"] is False


def test_temporal_summary_preserves_existing_gate_reasons() -> None:
    rows = []
    for frame, level, status in ((9, -0.2, "pass"), (10, -0.19, "reject"), (11, -0.21, "reject")):
        rows.append({
            "frame_index": frame,
            "geometry_available": True,
            "estimated_water_level_m": level,
            "water_area_m2": 2.0,
            "water_volume_m3": 0.2,
            "mean_depth_cm": 10.0,
            "max_depth_cm": 20.0,
            "camera_reprojection_iou": 0.95,
            "boundary_reprojection_p95_px": 4.0,
            "quality_status": status,
            "gate_reasons": [] if status == "pass" else ["boundary_reprojection_error_above_threshold"],
        })
    summary = pipeline.summarize_video_geometry(rows, 10)
    assert summary["quality_status_counts"] == {"pass": 1, "reject": 2}
    assert summary["gate_reason_counts"]["boundary_reprojection_error_above_threshold"] == 2
    assert summary["adjacent_absolute_water_level_change_m"]["count"] == 2


def test_prediction_module_has_no_ground_truth_loader_or_case_tuning() -> None:
    source = inspect.getsource(pipeline)
    for forbidden in (
        "load_camera_mask_ground_truth",
        "load_ground_truth_evaluation_inputs",
        "camera_water_mask_gt",
        "water_level_gt",
        "depth_map_gt",
        "nominal_depth_cm",
    ):
        assert forbidden not in source
