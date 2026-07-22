#!/usr/bin/env python3
"""Canonical water-state contract for Phase 2D-C-9 shadow integration."""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _assert_prediction_only(*documents: dict[str, Any]) -> None:
    for document in documents:
        if document.get("ground_truth_used") is True or document.get("ground_truth_used_for_prediction") is True:
            raise ValueError("Ground Truth provenance cannot enter canonical prediction state")


def build_canonical_water_state(
    geometry: dict[str, Any],
    candidate_gate: dict[str, Any],
    config: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build one deterministic, non-authoritative shadow-mode state record."""
    _assert_prediction_only(geometry, candidate_gate, provenance)
    if config.get("deployment_mode") != "shadow":
        raise ValueError("Phase 2D-C-9A only permits deployment_mode=shadow")
    if config.get("allow_downstream_in_shadow") is not False:
        raise ValueError("Shadow mode must block all downstream use")

    geometry_available = bool(geometry.get("geometry_available", False))
    candidate_status = str(candidate_gate.get("camera_visible_status", "reject"))
    global_status = str(candidate_gate.get("global_scene_status", "unavailable"))
    legacy_status = str(geometry.get("quality_status", "reject"))
    visible_valid = geometry_available and candidate_status == "pass"

    if not visible_valid:
        result_semantics = "unavailable"
        area_volume_semantics = "unavailable"
        measurement_status = "rejected_candidate" if geometry_available else "unavailable"
    elif global_status == "complete":
        result_semantics = "global_scene_estimate"
        area_volume_semantics = "complete_estimate"
        measurement_status = "complete_estimate"
    elif global_status == "partial":
        result_semantics = "camera_visible_estimate"
        area_volume_semantics = "observable_lower_bound"
        measurement_status = "camera_visible_estimate"
    else:
        result_semantics = "unavailable"
        area_volume_semantics = "unavailable"
        measurement_status = "unavailable"

    measurements = {
        "estimated_water_level_m": _finite_or_none(geometry.get("estimated_water_level_m")),
        "mean_depth_cm": _finite_or_none(geometry.get("mean_depth_cm")),
        "median_depth_cm": _finite_or_none(geometry.get("median_depth_cm")),
        "max_depth_cm": _finite_or_none(geometry.get("max_depth_cm")),
        "water_area_m2": _finite_or_none(geometry.get("water_area_m2")),
        "water_volume_m3": _finite_or_none(geometry.get("water_volume_m3")),
    }
    if not geometry_available:
        measurements = {key: None for key in measurements}

    reasons = list(candidate_gate.get("visible_reject_reasons", []))
    global_reasons = list(candidate_gate.get("global_scope_reasons", []))
    warnings = list(candidate_gate.get("warnings", []))
    return {
        "schema_version": str(config["schema_version"]),
        "stage": "S4_canonical_water_state_shadow",
        "identity": {
            "sample_id": str(candidate_gate.get("sample_id", geometry.get("sample_id", "unknown"))),
            "case_id": str(candidate_gate.get("case_id", "unknown")),
            "rain_level": str(candidate_gate.get("rain_level", "unknown")),
            "seed": int(candidate_gate.get("seed", -1)),
            "frame_index": int(geometry["frame_index"]),
        },
        "observation_scope": "camera_observable_region",
        "global_estimate_status": global_status if visible_valid else "unavailable",
        "observable_region_result_valid": visible_valid,
        "result_semantics": result_semantics,
        "area_volume_semantics": area_volume_semantics,
        "measurement_status": measurement_status,
        "measurements": measurements,
        "quality": {
            "deployment_mode": "shadow",
            "active_runtime_gate": str(config["active_runtime_gate"]),
            "legacy_runtime_gate": {
                "status": legacy_status,
                "reasons": list(geometry.get("gate_reasons", [])),
            },
            "candidate_gate": {
                "status": candidate_status,
                "global_scene_status": global_status,
                "visible_reject_reasons": reasons,
                "global_scope_reasons": global_reasons,
                "warnings": warnings,
                "boundary_metric_rejected_by_itself": bool(
                    candidate_gate.get("boundary_metric_rejected_by_itself", False)
                ),
            },
            "gate_status_disagreement": legacy_status != candidate_status,
            "candidate_gate_authoritative": False,
        },
        "provenance": {
            **provenance,
            "ground_truth_used": False,
            "real_world_validated": False,
            "synthetic_domain": True,
        },
        "authoritative": False,
        "eligible_for_downstream": False,
        "downstream_block_reason": "phase2d_c9a_shadow_mode_not_authoritative",
    }


def validate_canonical_water_state(state: dict[str, Any]) -> None:
    """Enforce semantics that prevent shadow candidates entering S5-S8."""
    if state.get("eligible_for_downstream") is not False or state.get("authoritative") is not False:
        raise ValueError("C9-A shadow state must never be authoritative or downstream-eligible")
    if state["quality"].get("deployment_mode") != "shadow":
        raise ValueError("C9-A state is not in shadow mode")
    if state["provenance"].get("ground_truth_used") is not False:
        raise ValueError("Canonical state has invalid Ground Truth provenance")
    if state["global_estimate_status"] == "partial":
        if state["result_semantics"] != "camera_visible_estimate":
            raise ValueError("Partial result must use camera_visible_estimate semantics")
        if state["area_volume_semantics"] != "observable_lower_bound":
            raise ValueError("Partial area/volume must be an observable lower bound")
    if state["observable_region_result_valid"] is False and state["result_semantics"] != "unavailable":
        raise ValueError("Invalid observable result cannot expose result semantics")
