"""Contract tests for the Phase 2D-C-8 quality-gate redesign protocol."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase2d_c8_quality_gate_protocol.yaml"


def _protocol() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[
        "phase2d_c8_quality_gate_protocol"
    ]


def test_protocol_is_design_only_and_does_not_enable_downstream() -> None:
    protocol = _protocol()
    assert protocol["status"] == "design_only"
    assert protocol["production_gate_modified"] is False
    assert protocol["prediction_modified"] is False
    assert protocol["authoritative_output_enabled"] is False
    assert protocol["downstream_enabled"] is False


def test_primary_project_target_is_three_centimeters() -> None:
    target = _protocol()["project_target"]
    assert target["primary_metric"] == "water_level_absolute_error_cm"
    assert target["maximum_water_level_absolute_error_cm"] == 3.0
    assert target["area_relative_error_limit"] is None
    assert target["volume_relative_error_limit"] is None


def test_visible_and_global_result_scopes_are_separate() -> None:
    scopes = _protocol()["result_scopes"]
    assert scopes["camera_visible_estimate"]["may_be_valid_when_global_scene_is_partial"] is True
    assert scopes["global_scene_estimate"]["requires_zero_unobservable_candidate_basins"] is True
    assert scopes["global_scene_estimate"]["requires_zero_ambiguous_candidate_basins"] is True


def test_ground_truth_and_evaluation_fields_are_forbidden_prediction_inputs() -> None:
    forbidden = set(_protocol()["forbidden_prediction_inputs"])
    assert {
        "camera_water_mask_gt",
        "dem_water_mask_gt",
        "depth_map_gt",
        "water_level_gt",
        "area_gt",
        "volume_gt",
        "evaluation_status",
        "evaluation_metrics",
    } <= forbidden


def test_boundary_metric_is_not_a_single_decision_or_direct_cm_conversion() -> None:
    boundary = _protocol()["decision_contract"]["boundary_metric"]
    assert boundary["role"] == "diagnostic_component_not_single_decision"
    assert boundary["current_runtime_threshold_px"] == 3.0
    assert boundary["redesigned_threshold_px"] is None
    assert boundary["direct_pixel_to_centimeter_conversion_allowed"] is False


def test_final_confirmation_is_disjoint_and_not_used_for_threshold_selection() -> None:
    confirmation = _protocol()["evidence_registry"]["final_confirmation"]
    assert confirmation["required"] is True
    assert confirmation["status"] == "data_not_yet_frozen"
    assert confirmation["required_matrix"]["seed"] == 303
    assert confirmation["required_matrix"]["minimum_sample_count"] == 12
    assert confirmation["must_not_be_used_to_select_thresholds"] is True
