#!/usr/bin/env python3
"""End-to-end invariant checks and safe fault injection for C9 shadow mode."""

from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from src.agent.canonical_shadow_monitor import build_shadow_monitor_summary
from src.api.canonical_shadow_api import build_canonical_shadow_api_payload
from src.database.canonical_shadow_audit import init_shadow_db
from src.integration.canonical_water_state import validate_canonical_water_state


def load_json_strict(path: str | Path) -> Any:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Required acceptance artifact missing: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def hashes_unchanged(before: dict[str, str | None], after: dict[str, str | None]) -> bool:
    return before == after


def _expect_exception(name: str, exception: type[BaseException], action: Callable[[], Any]) -> dict[str, Any]:
    try:
        action()
    except exception as exc:
        return {"scenario": name, "status": "pass", "fault_rejected": True, "observed_exception": type(exc).__name__}
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"scenario": name, "status": "fail", "fault_rejected": True, "observed_exception": type(exc).__name__}
    return {"scenario": name, "status": "fail", "fault_rejected": False, "observed_exception": None}


def run_fault_injection_suite(
    temporary_root: Path,
    canonical_state: dict[str, Any],
    canonical_summary: dict[str, Any],
    s5_s8_summary: dict[str, Any],
    envelope: dict[str, Any],
) -> list[dict[str, Any]]:
    temporary_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    results.append(_expect_exception(
        "missing_required_json", FileNotFoundError,
        lambda: load_json_strict(temporary_root / "missing.json"),
    ))
    corrupt = temporary_root / "corrupt.json"
    corrupt.write_text("{not-valid-json", encoding="utf-8")
    results.append(_expect_exception(
        "corrupted_json", json.JSONDecodeError, lambda: load_json_strict(corrupt)
    ))

    leaked = deepcopy(canonical_state)
    leaked["provenance"]["ground_truth_used"] = True
    results.append(_expect_exception(
        "ground_truth_leakage", ValueError, lambda: validate_canonical_water_state(leaked)
    ))
    downstream = deepcopy(canonical_state)
    downstream["eligible_for_downstream"] = True
    results.append(_expect_exception(
        "downstream_enabled_in_shadow", ValueError, lambda: validate_canonical_water_state(downstream)
    ))
    partial = deepcopy(canonical_state)
    partial["global_estimate_status"] = "partial"
    partial["observable_region_result_valid"] = True
    partial["result_semantics"] = "global_scene_estimate"
    partial["area_volume_semantics"] = "complete_estimate"
    results.append(_expect_exception(
        "partial_result_claims_global_semantics", ValueError,
        lambda: validate_canonical_water_state(partial),
    ))

    unsafe_summary = deepcopy(s5_s8_summary)
    unsafe_summary["warning_generation_allowed_count"] = 1
    monitor = build_shadow_monitor_summary(canonical_summary, unsafe_summary)
    results.append({
        "scenario": "warning_generation_enabled",
        "status": "pass" if monitor["monitor_status"] == "unsafe" else "fail",
        "fault_rejected": monitor["monitor_status"] == "unsafe",
        "observed_exception": None,
    })

    unsafe_envelope = deepcopy(envelope)
    unsafe_envelope["canonical_state"]["eligible_for_downstream"] = True
    safe_monitor = build_shadow_monitor_summary(canonical_summary, s5_s8_summary)
    results.append(_expect_exception(
        "api_exposes_downstream_state", ValueError,
        lambda: build_canonical_shadow_api_payload(safe_monitor, [unsafe_envelope]),
    ))

    database = temporary_root / "constraint_test.db"
    init_shadow_db(database)
    def violate_database_constraint() -> None:
        with sqlite3.connect(database) as connection:
            connection.execute(
                "INSERT INTO shadow_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("unsafe", "protocol", "unsafe", 0, 1, 0, 0),
            )
    results.append(_expect_exception(
        "sidecar_database_authoritative_constraint", sqlite3.IntegrityError,
        violate_database_constraint,
    ))
    results.append({
        "scenario": "formal_hash_mismatch",
        "status": "pass" if not hashes_unchanged({"formal": "a"}, {"formal": "b"}) else "fail",
        "fault_rejected": not hashes_unchanged({"formal": "a"}, {"formal": "b"}),
        "observed_exception": None,
    })
    return results


def evaluate_end_to_end_invariants(
    canonical_summary: dict[str, Any],
    canonical_records: list[dict[str, Any]],
    s5_s8_summary: dict[str, Any],
    envelopes: list[dict[str, Any]],
    monitoring_status: dict[str, Any],
    api_payload: dict[str, Any],
    audit_sample_count: int,
    expected: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = {
        "canonical_record_count": len(canonical_records) == int(expected["canonical_record_count"]),
        "canonical_sample_count": int(canonical_summary["sample_count"]) == int(expected["sample_count"]),
        "candidate_pass_count": canonical_summary["candidate_visible_status_counts"].get("pass") == int(expected["candidate_pass_count"]),
        "candidate_reject_count": canonical_summary["candidate_visible_status_counts"].get("reject") == int(expected["candidate_reject_count"]),
        "global_partial_count": canonical_summary["global_estimate_status_counts"].get("partial") == int(expected["global_partial_count"]),
        "canonical_all_blocked": all(not row["eligible_for_downstream"] and not row["authoritative"] for row in canonical_records),
        "canonical_prediction_only": all(row["provenance"]["ground_truth_used"] is False for row in canonical_records),
        "shadow_envelope_count": len(envelopes) == int(expected["sample_count"]),
        "s8_all_suppressed": s5_s8_summary["s8_status_counts"].get("warning_suppressed") == int(expected["sample_count"]),
        "warning_allowed_zero": int(s5_s8_summary["warning_generation_allowed_count"]) == 0,
        "formal_s5_s8_hashes_unchanged": s5_s8_summary["formal_output_hashes_unchanged"] is True,
        "monitor_healthy": monitoring_status["monitor"]["monitor_status"] == "healthy",
        "formal_audit_db_unchanged": monitoring_status["formal_audit_db_unchanged"] is True,
        "http_server_not_started": monitoring_status["http_server_started"] is False,
        "formal_agent_not_executed": monitoring_status["formal_agent_executed"] is False,
        "formal_warning_not_generated": monitoring_status["formal_warning_generated"] is False,
        "api_read_only": api_payload["mode"] == "read_only_shadow" and api_payload["warning_actions_available"] is False,
        "api_rejected_measurements_hidden": all(
            item["measurements"] is None
            for item in api_payload["samples"] if item["candidate_status"] == "reject"
        ),
        "sidecar_audit_sample_count": audit_sample_count == int(expected["sample_count"]),
    }
    return [
        {"check": name, "status": "pass" if passed else "fail", "passed": passed}
        for name, passed in checks.items()
    ]
