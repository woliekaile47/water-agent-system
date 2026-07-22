#!/usr/bin/env python3
"""Shadow-only S5-S8 adapters for canonical water state."""

from __future__ import annotations

from typing import Any

from src.integration.canonical_water_state import validate_canonical_water_state


def build_s5_shadow_input(state: dict[str, Any]) -> dict[str, Any]:
    """Map canonical values to the existing S5 field vocabulary without writing S5 outputs."""
    validate_canonical_water_state(state)
    valid = bool(state["observable_region_result_valid"])
    values = state["measurements"]
    required = ("mean_depth_cm", "max_depth_cm", "water_area_m2", "water_volume_m3")
    values_available = valid and all(values.get(key) is not None for key in required)
    return {
        "stage": "S5_shadow_contract_check",
        "status": "compatible_candidate" if values_available else "blocked",
        "source_schema_version": state["schema_version"],
        "measurement_status": state["measurement_status"],
        "result_semantics": state["result_semantics"],
        "area_volume_semantics": state["area_volume_semantics"],
        "mean_depth_cm": values.get("mean_depth_cm") if values_available else None,
        "median_depth_cm": values.get("median_depth_cm") if values_available else None,
        "max_depth_cm": values.get("max_depth_cm") if values_available else None,
        "water_area_m2": values.get("water_area_m2") if values_available else None,
        "water_volume_m3": values.get("water_volume_m3") if values_available else None,
        "eligible_for_s6_s8": False,
        "formal_s5_output_written": False,
    }


def build_s7_shadow_preflight(
    states: list[dict[str, Any]],
    fps: float,
    minimum_history_minutes: float,
) -> dict[str, Any]:
    """Check temporal readiness without executing the existing forecast implementation."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    for state in states:
        validate_canonical_water_state(state)
    frames = sorted(int(state["identity"]["frame_index"]) for state in states)
    duration_minutes = 0.0 if len(frames) < 2 else (frames[-1] - frames[0]) / fps / 60.0
    valid_count = sum(bool(state["observable_region_result_valid"]) for state in states)
    history_ready = duration_minutes >= minimum_history_minutes and valid_count == len(states)
    reasons: list[str] = []
    if duration_minutes < minimum_history_minutes:
        reasons.append("insufficient_canonical_history_duration")
    if valid_count != len(states):
        reasons.append("canonical_history_contains_rejected_frames")
    reasons.append("shadow_mode_forecast_execution_disabled")
    return {
        "stage": "S7_shadow_preflight",
        "status": "ready_but_disabled" if history_ready else "not_ready",
        "frame_count": len(states),
        "valid_frame_count": valid_count,
        "history_duration_minutes": float(duration_minutes),
        "minimum_history_minutes": float(minimum_history_minutes),
        "history_ready": history_ready,
        "forecast_executed": False,
        "forecast_results": [],
        "reasons": reasons,
        "eligible_for_s8": False,
    }


def build_s8_shadow_decision(state: dict[str, Any], s7_preflight: dict[str, Any]) -> dict[str, Any]:
    """Produce an explicit suppression record; never create a warning level or action."""
    validate_canonical_water_state(state)
    reasons = ["phase2d_c9b_shadow_mode_warning_suppressed"]
    if not state["observable_region_result_valid"]:
        reasons.append("canonical_measurement_rejected")
    if state["global_estimate_status"] == "partial":
        reasons.append("global_scene_estimate_partial")
    if not bool(s7_preflight.get("history_ready", False)):
        reasons.append("s7_history_not_ready")
    return {
        "stage": "S8_shadow_preflight",
        "status": "warning_suppressed",
        "warning_generation_allowed": False,
        "overall_warning_level": None,
        "action_suggestion": None,
        "forecast_results": [],
        "reasons": reasons,
        "formal_warning_output_written": False,
        "authoritative": False,
        "eligible_for_downstream": False,
    }


def build_s5_s8_shadow_envelope(
    latest_state: dict[str, Any],
    sequence_states: list[dict[str, Any]],
    fps: float,
    minimum_history_minutes: float,
) -> dict[str, Any]:
    """Build one auditable S5-S8 shadow envelope for a sample."""
    s5 = build_s5_shadow_input(latest_state)
    s6 = {
        "stage": "S6_shadow_context",
        "status": "not_executed",
        "weather_context_available": False,
        "weather_correction_applied": False,
        "reason": "C9-B validates the canonical contract without injecting mock weather",
    }
    s7 = build_s7_shadow_preflight(sequence_states, fps, minimum_history_minutes)
    s8 = build_s8_shadow_decision(latest_state, s7)
    return {
        "protocol_version": "phase2d_c9b_s5_s8_shadow_v1",
        "identity": latest_state["identity"],
        "canonical_state": latest_state,
        "s5_shadow_input": s5,
        "s6_shadow_context": s6,
        "s7_shadow_preflight": s7,
        "s8_shadow_decision": s8,
        "formal_pipeline_behavior_changed": False,
        "authoritative": False,
        "eligible_for_downstream": False,
    }
