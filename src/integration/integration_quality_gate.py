#!/usr/bin/env python3
"""Joint visual/geometry gate for Phase 2D-A prediction outputs."""

from __future__ import annotations

from typing import Any


UNKNOWN_SEMANTICS = "no_temporal_evidence_not_confirmed_dry"


def evaluate_integration_quality_gate(
    visual_gate: dict[str, Any], geometry_gate: dict[str, Any] | None,
    geometry_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    visual_status = str(visual_gate.get("status", "reject"))
    geometry_status = "unavailable" if geometry_gate is None else str(geometry_gate.get("status", "reject"))
    reasons = [f"visual:{reason}" for reason in visual_gate.get("reasons", [])]
    if geometry_failure is not None or geometry_gate is None:
        failure_type = (geometry_failure or {}).get("failure_type", "geometry_unavailable")
        reasons.append(f"geometry:{failure_type}")
        status = "reject"
        global_status = "unavailable"
        observable_valid = False
        result_semantics = "unavailable"
        area_semantics = "unavailable"
    else:
        reasons.extend(f"geometry:{reason}" for reason in geometry_gate.get("reasons", []))
        global_status = str(geometry_gate.get("global_estimate_status", "unavailable"))
        observable_valid = bool(geometry_gate.get("observable_region_result_valid", False))
        result_semantics = str(geometry_gate.get("result_semantics", "observable_region_estimate"))
        area_semantics = str(geometry_gate.get("area_volume_semantics", "observable_lower_bound"))
        if visual_status == "reject" or geometry_status == "reject":
            status = "reject"
        elif visual_status == "partial" or geometry_status == "partial" or global_status != "complete":
            status = "partial"
        elif visual_status == "pass" and geometry_status == "pass" and global_status == "complete":
            status = "pass"
        else:
            status = "partial"
    downstream_reason = (
        "integration_quality_gate_reject" if status == "reject"
        else "phase2d_a_prediction_experiment_not_connected_to_s5_s8"
    )
    candidate_values_available = geometry_failure is None and geometry_gate is not None
    if not candidate_values_available:
        measurement_status = "unavailable"
        authoritative_measurement_available = False
        authoritative_area_volume_semantics = "unavailable"
    elif status == "pass":
        measurement_status = "complete_estimate"
        authoritative_measurement_available = True
        authoritative_area_volume_semantics = "complete_estimate"
    elif status == "partial" and observable_valid:
        measurement_status = "observable_region_estimate"
        authoritative_measurement_available = True
        authoritative_area_volume_semantics = "observable_lower_bound"
    else:
        measurement_status = "rejected_candidate"
        authoritative_measurement_available = False
        authoritative_area_volume_semantics = "unavailable"
    return {
        "status": status, "reasons": reasons,
        "visual_gate_status": visual_status, "geometry_gate_status": geometry_status,
        "global_estimate_status": global_status,
        "observable_region_result_valid": observable_valid,
        "result_semantics": result_semantics,
        "area_volume_semantics": area_semantics,
        "measurement_status": measurement_status,
        "candidate_values_available": candidate_values_available,
        "authoritative_measurement_available": authoritative_measurement_available,
        "authoritative_area_volume_semantics": authoritative_area_volume_semantics,
        "unknown_region_semantics": UNKNOWN_SEMANTICS,
        "synthetic_domain": True, "real_world_validated": False,
        "eligible_for_downstream": False,
        "downstream_block_reason": downstream_reason,
    }
