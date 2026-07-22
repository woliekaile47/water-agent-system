#!/usr/bin/env python3
"""Fail-closed routing contract for simulation-only end-to-end execution."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.integration.canonical_water_state import validate_canonical_water_state


def _stable_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_simulation_runtime_config(config: dict[str, Any]) -> None:
    """Reject any configuration that can escape the simulation-only boundary."""
    if config.get("runtime_mode") != "simulation_e2e":
        raise ValueError("C11-A only permits runtime_mode=simulation_e2e")
    if config.get("data_domain") != "simulation":
        raise ValueError("C11-A only permits data_domain=simulation")
    required_false = (
        "allow_formal_output_writes",
        "allow_real_warning",
        "allow_external_notification",
        "allow_real_device_actions",
        "allow_simulation_action_execution",
    )
    unsafe = [key for key in required_false if config.get(key) is not False]
    if unsafe:
        raise ValueError(f"Unsafe simulation runtime flags: {', '.join(unsafe)}")
    if config.get("warning_action_mode") != "simulation_record_only":
        raise ValueError("warning_action_mode must remain simulation_record_only")
    required_true = (
        "allow_simulation_s5_s6_routing",
        "allow_simulation_global_s7_s8_routing",
    )
    disabled = [key for key in required_true if config.get(key) is not True]
    if disabled:
        raise ValueError(f"Required C11-A routing flags are disabled: {', '.join(disabled)}")


def build_simulation_runtime_envelope(
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Convert one frozen C9 state into an isolated simulation routing decision."""
    validate_simulation_runtime_config(config)
    validate_canonical_water_state(state)
    provenance = state.get("provenance", {})
    if provenance.get("ground_truth_used") is not False:
        raise ValueError("Ground Truth provenance cannot enter simulation runtime routing")
    if provenance.get("synthetic_domain") is not True:
        raise ValueError("C11-A requires a synthetic-domain canonical state")

    visible_valid = bool(state.get("observable_region_result_valid", False))
    global_status = str(state.get("global_estimate_status", "unavailable"))
    complete = visible_valid and global_status == "complete"
    partial = visible_valid and global_status == "partial"
    simulation_eligible = complete or partial

    if complete:
        routing_status = "eligible_complete"
        block_reasons: list[str] = []
    elif partial:
        routing_status = "eligible_camera_visible_only"
        block_reasons = ["global_scene_estimate_partial"]
    else:
        routing_status = "blocked"
        block_reasons = ["canonical_candidate_not_valid"]

    measurements = dict(state.get("measurements", {})) if simulation_eligible else {
        key: None for key in state.get("measurements", {})
    }
    return {
        "protocol_version": str(config["protocol_version"]),
        "stage": "C11A_simulation_e2e_runtime_routing",
        "identity": dict(state["identity"]),
        "runtime_mode": "simulation_e2e",
        "data_domain": "simulation",
        "source_contract": {
            "schema_version": state["schema_version"],
            "deployment_mode": state["quality"]["deployment_mode"],
            "canonical_state_sha256": _stable_sha256(state),
            "authoritative": state["authoritative"],
            "eligible_for_downstream": state["eligible_for_downstream"],
            "ground_truth_used": False,
        },
        "result_semantics": state["result_semantics"] if simulation_eligible else "unavailable",
        "area_volume_semantics": state["area_volume_semantics"] if simulation_eligible else "unavailable",
        "global_estimate_status": global_status if simulation_eligible else "unavailable",
        "measurements": measurements,
        "simulation_routing": {
            "status": routing_status,
            "eligible_for_simulation_pipeline": simulation_eligible,
            "eligible_for_simulation_s5_s6": simulation_eligible,
            "eligible_for_simulation_global_s7_s8": complete,
            "eligible_for_simulation_warning_evaluation": complete,
            "block_reasons": block_reasons,
        },
        "safety": {
            "eligible_for_real_warning": False,
            "warning_action_mode": "simulation_record_only",
            "external_notification_allowed": False,
            "real_device_action_allowed": False,
            "formal_output_writes_allowed": False,
            "simulation_action_execution_allowed": False,
        },
        "authoritative": False,
        "eligible_for_downstream": False,
        "ground_truth_used": False,
    }


def validate_simulation_runtime_envelope(envelope: dict[str, Any]) -> None:
    """Enforce routing semantics and the permanent real-world action block."""
    if envelope.get("runtime_mode") != "simulation_e2e" or envelope.get("data_domain") != "simulation":
        raise ValueError("Envelope is outside the simulation_e2e runtime domain")
    if envelope.get("authoritative") is not False or envelope.get("eligible_for_downstream") is not False:
        raise ValueError("Simulation envelope must remain non-authoritative")
    if envelope.get("ground_truth_used") is not False:
        raise ValueError("Simulation runtime envelope has invalid GT provenance")
    source = envelope["source_contract"]
    if source.get("authoritative") is not False or source.get("eligible_for_downstream") is not False:
        raise ValueError("Source canonical state escaped the C9 shadow boundary")
    if source.get("ground_truth_used") is not False:
        raise ValueError("Source canonical contract has invalid GT provenance")
    safety = envelope["safety"]
    if any(
        safety.get(key) is not False
        for key in (
            "eligible_for_real_warning",
            "external_notification_allowed",
            "real_device_action_allowed",
            "formal_output_writes_allowed",
            "simulation_action_execution_allowed",
        )
    ):
        raise ValueError("Simulation envelope attempted a real or formal side effect")
    if safety.get("warning_action_mode") != "simulation_record_only":
        raise ValueError("Invalid simulation warning action mode")

    routing = envelope["simulation_routing"]
    status = envelope["global_estimate_status"]
    if status == "complete":
        if any(
            routing.get(key) is not True
            for key in (
                "eligible_for_simulation_pipeline",
                "eligible_for_simulation_s5_s6",
                "eligible_for_simulation_global_s7_s8",
                "eligible_for_simulation_warning_evaluation",
            )
        ):
            raise ValueError("Complete simulation estimate must be globally routable")
        if envelope.get("area_volume_semantics") != "complete_estimate":
            raise ValueError("Complete result must carry complete area/volume semantics")
    elif status == "partial":
        if routing.get("eligible_for_simulation_pipeline") is not True or routing.get(
            "eligible_for_simulation_s5_s6"
        ) is not True:
            raise ValueError("Partial visible estimate should remain simulation-routable")
        if routing.get("eligible_for_simulation_global_s7_s8") is not False or routing.get(
            "eligible_for_simulation_warning_evaluation"
        ) is not False:
            raise ValueError("Partial estimate cannot enter global S7-S8 simulation routing")
        if envelope.get("result_semantics") != "camera_visible_estimate":
            raise ValueError("Partial estimate must retain camera-visible semantics")
        if envelope.get("area_volume_semantics") != "observable_lower_bound":
            raise ValueError("Partial area/volume must remain a lower bound")
    else:
        if any(
            routing.get(key) is not False
            for key in (
                "eligible_for_simulation_pipeline",
                "eligible_for_simulation_s5_s6",
                "eligible_for_simulation_global_s7_s8",
                "eligible_for_simulation_warning_evaluation",
            )
        ):
            raise ValueError("Unavailable result cannot enter any simulation stage")
        if any(value is not None for value in envelope.get("measurements", {}).values()):
            raise ValueError("Blocked simulation result cannot expose measurements")
