"""Tests for the offline Phase 2D-C-8 candidate quality gate."""

from copy import deepcopy
from pathlib import Path

import yaml

from src.evaluation.phase2d_c8_candidate_quality_gate import evaluate_candidate_gate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (ROOT / "configs" / "phase2d_c8_candidate_quality_gate.yaml").read_text(encoding="utf-8")
)["phase2d_c8_candidate_quality_gate"]


def frame() -> dict:
    return {
        "frame_index": 99,
        "geometry_available": True,
        "selected_basin_count": 1,
        "estimated_water_level_m": -0.2,
        "max_depth_cm": 20.0,
        "shoreline_intersection_success_rate": 1.0,
        "seed_ray_success_ratio": 1.0,
        "filtered_shoreline_sample_count": 128,
        "filtered_shoreline_mad_m": 0.004,
        "filtered_shoreline_iqr_m": 0.009,
        "camera_reprojection_iou": 0.95,
        "outer_boundary_reprojection_p95_px": 2.0,
        "unobserved_candidate_basin_count": 0,
        "ambiguous_candidate_basin_count": 0,
    }


def sequence() -> dict:
    return {
        "estimated_water_level_m": {"std": 0.002},
        "adjacent_absolute_water_level_change_m": {"p95": 0.004},
        "water_area_m2": {"coefficient_of_variation": 0.02},
        "water_volume_m3": {"coefficient_of_variation": 0.03},
    }


def test_stable_visible_and_complete_case_passes() -> None:
    result = evaluate_candidate_gate(frame(), sequence(), CONFIG)
    assert result["camera_visible_status"] == "pass"
    assert result["global_scene_status"] == "complete"
    assert result["result_semantics"] == "global_scene_estimate"
    assert result["eligible_for_downstream"] is False


def test_boundary_warning_does_not_reject_by_itself() -> None:
    candidate = frame()
    candidate["outer_boundary_reprojection_p95_px"] = 20.0
    result = evaluate_candidate_gate(candidate, sequence(), CONFIG)
    assert result["camera_visible_status"] == "pass"
    assert result["boundary_metric_rejected_by_itself"] is False
    assert "outer_boundary_reprojection_p95_above_advisory_threshold" in result["warnings"]


def test_low_camera_reprojection_rejects_visible_estimate() -> None:
    candidate = frame()
    candidate["camera_reprojection_iou"] = 0.7
    result = evaluate_candidate_gate(candidate, sequence(), CONFIG)
    assert result["camera_visible_status"] == "reject"
    assert result["global_scene_status"] == "unavailable"


def test_temporal_instability_rejects_visible_estimate() -> None:
    unstable = deepcopy(sequence())
    unstable["water_volume_m3"]["coefficient_of_variation"] = 0.31
    result = evaluate_candidate_gate(frame(), unstable, CONFIG)
    assert result["camera_visible_status"] == "reject"
    assert "water_volume_cv_above_candidate_threshold" in result["temporal_reasons"]


def test_unobservable_basin_preserves_visible_but_makes_global_partial() -> None:
    candidate = frame()
    candidate["unobserved_candidate_basin_count"] = 1
    candidate["ambiguous_candidate_basin_count"] = 1
    result = evaluate_candidate_gate(candidate, sequence(), CONFIG)
    assert result["camera_visible_status"] == "pass"
    assert result["global_scene_status"] == "partial"
    assert result["result_semantics"] == "camera_visible_estimate"


def test_empty_dry_candidate_cannot_create_false_water_measurement() -> None:
    candidate = frame()
    candidate["geometry_available"] = False
    candidate["selected_basin_count"] = 0
    candidate["estimated_water_level_m"] = None
    result = evaluate_candidate_gate(candidate, sequence(), CONFIG)
    assert result["camera_visible_status"] == "reject"
    assert result["global_scene_status"] == "unavailable"
    assert result["eligible_for_downstream"] is False


def test_candidate_api_has_no_ground_truth_argument() -> None:
    assert evaluate_candidate_gate.__code__.co_varnames[:3] == (
        "frame_metrics", "sequence_metrics", "config"
    )
