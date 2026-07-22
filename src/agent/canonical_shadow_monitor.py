#!/usr/bin/env python3
"""Read-only Agent sidecar monitor for C9 canonical/shadow artifacts."""

from __future__ import annotations

from typing import Any


def build_shadow_monitor_summary(
    canonical_summary: dict[str, Any],
    s5_s8_summary: dict[str, Any],
) -> dict[str, Any]:
    if canonical_summary.get("ground_truth_used") is not False:
        raise ValueError("Canonical monitor input has invalid GT provenance")
    if s5_s8_summary.get("ground_truth_used") is not False:
        raise ValueError("S5-S8 monitor input has invalid GT provenance")
    safe = (
        int(canonical_summary.get("downstream_eligible_count", -1)) == 0
        and int(canonical_summary.get("authoritative_count", -1)) == 0
        and int(s5_s8_summary.get("warning_generation_allowed_count", -1)) == 0
        and int(s5_s8_summary.get("downstream_eligible_count", -1)) == 0
        and s5_s8_summary.get("formal_output_hashes_unchanged") is True
        and s5_s8_summary.get("formal_s5_s8_executed") is False
    )
    return {
        "monitor_mode": "canonical_shadow_sidecar",
        "monitor_status": "healthy" if safe else "unsafe",
        "canonical_record_count": int(canonical_summary["record_count"]),
        "canonical_sample_count": int(canonical_summary["sample_count"]),
        "candidate_visible_status_counts": canonical_summary["candidate_visible_status_counts"],
        "global_estimate_status_counts": canonical_summary["global_estimate_status_counts"],
        "legacy_candidate_status_matrix": canonical_summary["legacy_candidate_status_matrix"],
        "s5_status_counts": s5_s8_summary["s5_status_counts"],
        "s7_status_counts": s5_s8_summary["s7_status_counts"],
        "s8_status_counts": s5_s8_summary["s8_status_counts"],
        "formal_outputs_unchanged": bool(s5_s8_summary["formal_output_hashes_unchanged"]),
        "formal_agent_pipeline_modified": False,
        "formal_warning_generated": False,
        "authoritative": False,
        "eligible_for_downstream": False,
    }
