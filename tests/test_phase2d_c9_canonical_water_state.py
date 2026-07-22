"""Tests for the Phase 2D-C-9A canonical water-state shadow contract."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.integration.canonical_water_state import (
    build_canonical_water_state,
    validate_canonical_water_state,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (ROOT / "configs" / "phase2d_c9_canonical_water_state.yaml").read_text(encoding="utf-8")
)["phase2d_c9_canonical_water_state"]


def geometry() -> dict:
    return {
        "sample_id": "sample",
        "frame_index": 149,
        "geometry_available": True,
        "quality_status": "reject",
        "gate_reasons": ["boundary_reprojection_error_above_threshold"],
        "estimated_water_level_m": -0.2,
        "mean_depth_cm": 8.0,
        "median_depth_cm": 7.0,
        "max_depth_cm": 20.0,
        "water_area_m2": 7.0,
        "water_volume_m3": 0.6,
        "ground_truth_used": False,
    }


def candidate() -> dict:
    return {
        "sample_id": "sample",
        "case_id": "sim_water_20cm_001",
        "rain_level": "moderate",
        "seed": 303,
        "frame_index": 149,
        "camera_visible_status": "pass",
        "global_scene_status": "complete",
        "visible_reject_reasons": [],
        "global_scope_reasons": [],
        "warnings": ["outer_boundary_reprojection_p95_above_advisory_threshold"],
        "boundary_metric_rejected_by_itself": False,
        "ground_truth_used": False,
    }


def test_complete_candidate_is_still_blocked_in_shadow_mode() -> None:
    state = build_canonical_water_state(geometry(), candidate(), CONFIG, {"ground_truth_used": False})
    validate_canonical_water_state(state)
    assert state["result_semantics"] == "global_scene_estimate"
    assert state["area_volume_semantics"] == "complete_estimate"
    assert state["quality"]["gate_status_disagreement"] is True
    assert state["eligible_for_downstream"] is False
    assert state["authoritative"] is False


def test_partial_candidate_is_visible_lower_bound() -> None:
    decision = candidate()
    decision["global_scene_status"] = "partial"
    decision["global_scope_reasons"] = ["candidate_basin_outside_camera_coverage"]
    state = build_canonical_water_state(geometry(), decision, CONFIG, {"ground_truth_used": False})
    validate_canonical_water_state(state)
    assert state["result_semantics"] == "camera_visible_estimate"
    assert state["area_volume_semantics"] == "observable_lower_bound"


def test_rejected_candidate_has_no_result_semantics() -> None:
    decision = candidate()
    decision["camera_visible_status"] = "reject"
    decision["global_scene_status"] = "unavailable"
    decision["visible_reject_reasons"] = ["camera_reprojection_iou_below_candidate_threshold"]
    state = build_canonical_water_state(geometry(), decision, CONFIG, {"ground_truth_used": False})
    assert state["observable_region_result_valid"] is False
    assert state["result_semantics"] == "unavailable"
    assert state["measurement_status"] == "rejected_candidate"


def test_unavailable_geometry_clears_measurements() -> None:
    unavailable = geometry()
    unavailable["geometry_available"] = False
    state = build_canonical_water_state(unavailable, candidate(), CONFIG, {"ground_truth_used": False})
    assert all(value is None for value in state["measurements"].values())
    assert state["measurement_status"] == "unavailable"


def test_ground_truth_provenance_is_rejected() -> None:
    with pytest.raises(ValueError, match="Ground Truth"):
        build_canonical_water_state(geometry(), candidate(), CONFIG, {"ground_truth_used": True})


def test_non_shadow_or_downstream_enabled_config_is_rejected() -> None:
    active = deepcopy(CONFIG)
    active["deployment_mode"] = "active"
    with pytest.raises(ValueError, match="shadow"):
        build_canonical_water_state(geometry(), candidate(), active, {"ground_truth_used": False})
    unsafe = deepcopy(CONFIG)
    unsafe["allow_downstream_in_shadow"] = True
    with pytest.raises(ValueError, match="block"):
        build_canonical_water_state(geometry(), candidate(), unsafe, {"ground_truth_used": False})


def test_same_input_is_deterministic() -> None:
    first = build_canonical_water_state(geometry(), candidate(), CONFIG, {"ground_truth_used": False})
    second = build_canonical_water_state(geometry(), candidate(), CONFIG, {"ground_truth_used": False})
    assert first == second
