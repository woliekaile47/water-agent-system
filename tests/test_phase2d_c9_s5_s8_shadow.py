"""Tests for the C9-B S5-S8 shadow adapter."""

from copy import deepcopy
from pathlib import Path

import yaml

from src.integration.canonical_s5_s8_shadow import (
    build_s5_s8_shadow_envelope,
    build_s5_shadow_input,
    build_s7_shadow_preflight,
    build_s8_shadow_decision,
)
from src.integration.canonical_water_state import build_canonical_water_state


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = yaml.safe_load(
    (ROOT / "configs" / "phase2d_c9_canonical_water_state.yaml").read_text(encoding="utf-8")
)["phase2d_c9_canonical_water_state"]


def state(global_status: str = "complete", candidate_status: str = "pass", frame: int = 149):
    geometry = {
        "sample_id": "sample", "frame_index": frame, "geometry_available": True,
        "quality_status": "reject", "gate_reasons": ["boundary"],
        "estimated_water_level_m": -0.2, "mean_depth_cm": 8.0,
        "median_depth_cm": 7.0, "max_depth_cm": 20.0,
        "water_area_m2": 7.0, "water_volume_m3": 0.6,
        "ground_truth_used": False,
    }
    candidate = {
        "sample_id": "sample", "case_id": "sim_water_20cm_001",
        "rain_level": "moderate", "seed": 303, "frame_index": frame,
        "camera_visible_status": candidate_status, "global_scene_status": global_status,
        "visible_reject_reasons": [] if candidate_status == "pass" else ["low_iou"],
        "global_scope_reasons": [] if global_status != "partial" else ["outside_camera"],
        "warnings": [], "ground_truth_used": False,
    }
    return build_canonical_water_state(
        geometry, candidate, CANONICAL_CONFIG, {"ground_truth_used": False}
    )


def test_complete_and_partial_s5_field_compatibility() -> None:
    complete = build_s5_shadow_input(state())
    partial = build_s5_shadow_input(state(global_status="partial"))
    assert complete["status"] == "compatible_candidate"
    assert complete["area_volume_semantics"] == "complete_estimate"
    assert partial["status"] == "compatible_candidate"
    assert partial["area_volume_semantics"] == "observable_lower_bound"
    assert partial["eligible_for_s6_s8"] is False


def test_rejected_candidate_cannot_supply_s5_values() -> None:
    rejected = build_s5_shadow_input(state(global_status="unavailable", candidate_status="reject"))
    assert rejected["status"] == "blocked"
    assert rejected["water_area_m2"] is None
    assert rejected["formal_s5_output_written"] is False


def test_two_second_history_does_not_fake_a_ten_minute_forecast() -> None:
    states = [state(frame=frame) for frame in range(129, 170)]
    result = build_s7_shadow_preflight(states, fps=20.0, minimum_history_minutes=10.0)
    assert result["history_duration_minutes"] < 0.04
    assert result["history_ready"] is False
    assert result["forecast_executed"] is False
    assert result["forecast_results"] == []


def test_s8_is_always_suppressed_in_shadow_mode() -> None:
    current = state(global_status="partial")
    preflight = build_s7_shadow_preflight([current], fps=20.0, minimum_history_minutes=10.0)
    result = build_s8_shadow_decision(current, preflight)
    assert result["status"] == "warning_suppressed"
    assert result["warning_generation_allowed"] is False
    assert result["overall_warning_level"] is None
    assert "global_scene_estimate_partial" in result["reasons"]


def test_envelope_never_changes_formal_behavior() -> None:
    current = state()
    envelope = build_s5_s8_shadow_envelope(current, [current], 20.0, 10.0)
    assert envelope["formal_pipeline_behavior_changed"] is False
    assert envelope["authoritative"] is False
    assert envelope["eligible_for_downstream"] is False


def test_source_has_no_formal_stage_calls() -> None:
    source = (ROOT / "scripts" / "run_phase2d_c9_s5_s8_shadow.py").read_text(encoding="utf-8")
    for forbidden in (
        "calculate_area_volume(", "compute_weather_correction(",
        "deterministic_forecast(", "generate_warning_decision(",
    ):
        assert forbidden not in source
