"""Tests for the C11-A simulation-only runtime routing contract."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.integration.canonical_water_state import build_canonical_water_state
from src.integration.simulation_e2e_runtime import (
    build_simulation_runtime_envelope,
    validate_simulation_runtime_config,
    validate_simulation_runtime_envelope,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = yaml.safe_load(
    (ROOT / "configs" / "phase2d_c9_canonical_water_state.yaml").read_text(encoding="utf-8")
)["phase2d_c9_canonical_water_state"]
RUNTIME_CONFIG = yaml.safe_load(
    (ROOT / "configs" / "phase2d_c11_simulation_e2e_runtime.yaml").read_text(encoding="utf-8")
)["phase2d_c11_simulation_e2e_runtime"]


def state(global_status: str = "complete", candidate_status: str = "pass") -> dict:
    geometry = {
        "sample_id": "sample", "frame_index": 149, "geometry_available": True,
        "quality_status": "reject", "gate_reasons": ["legacy_boundary"],
        "estimated_water_level_m": -0.2, "mean_depth_cm": 8.0,
        "median_depth_cm": 7.0, "max_depth_cm": 20.0,
        "water_area_m2": 7.0, "water_volume_m3": 0.6,
        "ground_truth_used": False,
    }
    candidate = {
        "sample_id": "sample", "case_id": "sim_water_20cm_001",
        "rain_level": "moderate", "seed": 303, "frame_index": 149,
        "camera_visible_status": candidate_status, "global_scene_status": global_status,
        "visible_reject_reasons": [] if candidate_status == "pass" else ["low_iou"],
        "global_scope_reasons": [] if global_status != "partial" else ["outside_camera"],
        "warnings": [], "ground_truth_used": False,
    }
    return build_canonical_water_state(
        geometry,
        candidate,
        CANONICAL_CONFIG,
        {"ground_truth_used": False, "synthetic_domain": True},
    )


def test_complete_result_enters_global_simulation_route_only() -> None:
    result = build_simulation_runtime_envelope(state(), RUNTIME_CONFIG)
    validate_simulation_runtime_envelope(result)
    assert result["simulation_routing"]["eligible_for_simulation_pipeline"] is True
    assert result["simulation_routing"]["eligible_for_simulation_global_s7_s8"] is True
    assert result["area_volume_semantics"] == "complete_estimate"
    assert result["eligible_for_downstream"] is False


def test_partial_result_is_visible_lower_bound_not_global() -> None:
    result = build_simulation_runtime_envelope(state(global_status="partial"), RUNTIME_CONFIG)
    validate_simulation_runtime_envelope(result)
    assert result["simulation_routing"]["eligible_for_simulation_s5_s6"] is True
    assert result["simulation_routing"]["eligible_for_simulation_global_s7_s8"] is False
    assert result["result_semantics"] == "camera_visible_estimate"
    assert result["area_volume_semantics"] == "observable_lower_bound"


def test_rejected_result_is_blocked_and_measurements_are_hidden() -> None:
    result = build_simulation_runtime_envelope(
        state(global_status="unavailable", candidate_status="reject"), RUNTIME_CONFIG
    )
    validate_simulation_runtime_envelope(result)
    assert result["simulation_routing"]["status"] == "blocked"
    assert result["simulation_routing"]["eligible_for_simulation_pipeline"] is False
    assert all(value is None for value in result["measurements"].values())


def test_real_warning_notification_and_devices_are_always_blocked() -> None:
    safety = build_simulation_runtime_envelope(state(), RUNTIME_CONFIG)["safety"]
    assert safety == {
        "eligible_for_real_warning": False,
        "warning_action_mode": "simulation_record_only",
        "external_notification_allowed": False,
        "real_device_action_allowed": False,
        "formal_output_writes_allowed": False,
        "simulation_action_execution_allowed": False,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("runtime_mode", "production"),
        ("data_domain", "real"),
        ("allow_formal_output_writes", True),
        ("allow_real_warning", True),
        ("allow_external_notification", True),
        ("allow_real_device_actions", True),
        ("allow_simulation_action_execution", True),
        ("warning_action_mode", "send_warning"),
        ("allow_simulation_s5_s6_routing", False),
        ("allow_simulation_global_s7_s8_routing", False),
    ],
)
def test_unsafe_runtime_configuration_is_rejected(key: str, value: object) -> None:
    config = deepcopy(RUNTIME_CONFIG)
    config[key] = value
    with pytest.raises(ValueError):
        validate_simulation_runtime_config(config)


def test_non_synthetic_or_gt_tainted_source_is_rejected() -> None:
    real = state()
    real["provenance"]["synthetic_domain"] = False
    with pytest.raises(ValueError, match="synthetic"):
        build_simulation_runtime_envelope(real, RUNTIME_CONFIG)
    tainted = state()
    tainted["provenance"]["ground_truth_used"] = True
    with pytest.raises(ValueError, match="Ground Truth"):
        build_simulation_runtime_envelope(tainted, RUNTIME_CONFIG)


def test_same_input_is_deterministic() -> None:
    first = build_simulation_runtime_envelope(state(), RUNTIME_CONFIG)
    second = build_simulation_runtime_envelope(state(), RUNTIME_CONFIG)
    assert first == second


def test_builder_has_no_formal_pipeline_or_notification_calls() -> None:
    source = (ROOT / "scripts" / "build_phase2d_c11_simulation_e2e_runtime.py").read_text(encoding="utf-8")
    for forbidden in (
        "calculate_area_volume(", "compute_weather_correction(",
        "deterministic_forecast(", "generate_warning_decision(",
        "requests.post(", "send_sms(", "send_email(",
    ):
        assert forbidden not in source
