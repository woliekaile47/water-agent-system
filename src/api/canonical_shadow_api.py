#!/usr/bin/env python3
"""Framework-neutral read-only API payload for C9 shadow monitoring."""

from __future__ import annotations

from typing import Any


def build_canonical_shadow_api_payload(
    monitor: dict[str, Any],
    envelopes: list[dict[str, Any]],
) -> dict[str, Any]:
    if monitor.get("authoritative") is not False or monitor.get("eligible_for_downstream") is not False:
        raise ValueError("API refuses authoritative/downstream-enabled shadow monitor data")
    samples = []
    for envelope in envelopes:
        state = envelope["canonical_state"]
        if state.get("eligible_for_downstream") is not False:
            raise ValueError("API refuses downstream-eligible canonical state")
        samples.append({
            "sample_id": state["identity"]["sample_id"],
            "case_id": state["identity"]["case_id"],
            "frame_index": state["identity"]["frame_index"],
            "candidate_status": state["quality"]["candidate_gate"]["status"],
            "global_estimate_status": state["global_estimate_status"],
            "result_semantics": state["result_semantics"],
            "area_volume_semantics": state["area_volume_semantics"],
            "measurements": state["measurements"] if state["observable_region_result_valid"] else None,
            "s5_shadow_status": envelope["s5_shadow_input"]["status"],
            "s7_shadow_status": envelope["s7_shadow_preflight"]["status"],
            "s8_shadow_status": envelope["s8_shadow_decision"]["status"],
            "warning_level": None,
            "authoritative": False,
            "eligible_for_downstream": False,
        })
    return {
        "api_schema_version": "canonical_shadow_read_api_v1",
        "mode": "read_only_shadow",
        "monitor_status": monitor["monitor_status"],
        "summary": monitor,
        "samples": samples,
        "warning_actions_available": False,
        "authoritative": False,
        "eligible_for_downstream": False,
        "note": "Framework-neutral payload; an HTTP server is not started in C9-C.",
    }
