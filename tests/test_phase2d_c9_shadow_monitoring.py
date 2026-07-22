"""Tests for C9-C Agent/DB/API shadow monitoring."""

from pathlib import Path

from src.agent.canonical_shadow_monitor import build_shadow_monitor_summary
from src.api.canonical_shadow_api import build_canonical_shadow_api_payload
from src.database.canonical_shadow_audit import read_shadow_audit, write_shadow_audit


def summaries():
    canonical = {
        "record_count": 2, "sample_count": 1, "ground_truth_used": False,
        "downstream_eligible_count": 0, "authoritative_count": 0,
        "candidate_visible_status_counts": {"pass": 2},
        "global_estimate_status_counts": {"complete": 2},
        "legacy_candidate_status_matrix": {"legacy_reject__candidate_pass": 2},
    }
    downstream = {
        "ground_truth_used": False, "warning_generation_allowed_count": 0,
        "downstream_eligible_count": 0, "formal_output_hashes_unchanged": True,
        "formal_s5_s8_executed": False, "s5_status_counts": {"compatible_candidate": 1},
        "s7_status_counts": {"not_ready": 1}, "s8_status_counts": {"warning_suppressed": 1},
    }
    return canonical, downstream


def envelope():
    state = {
        "identity": {"sample_id": "sample", "case_id": "case", "frame_index": 169},
        "global_estimate_status": "complete", "result_semantics": "global_scene_estimate",
        "area_volume_semantics": "complete_estimate", "observable_region_result_valid": True,
        "measurements": {"mean_depth_cm": 8.0, "max_depth_cm": 20.0, "water_area_m2": 7.0, "water_volume_m3": 0.6},
        "quality": {"candidate_gate": {"status": "pass"}},
        "eligible_for_downstream": False, "authoritative": False,
    }
    return {
        "canonical_state": state,
        "s5_shadow_input": {"status": "compatible_candidate"},
        "s7_shadow_preflight": {"status": "not_ready"},
        "s8_shadow_decision": {"status": "warning_suppressed"},
        "eligible_for_downstream": False, "authoritative": False,
    }


def test_agent_sidecar_reports_healthy_only_when_all_safety_checks_hold() -> None:
    canonical, downstream = summaries()
    monitor = build_shadow_monitor_summary(canonical, downstream)
    assert monitor["monitor_status"] == "healthy"
    assert monitor["formal_agent_pipeline_modified"] is False
    downstream["warning_generation_allowed_count"] = 1
    assert build_shadow_monitor_summary(canonical, downstream)["monitor_status"] == "unsafe"


def test_api_payload_is_read_only_and_hides_rejected_measurements() -> None:
    canonical, downstream = summaries()
    monitor = build_shadow_monitor_summary(canonical, downstream)
    accepted = envelope()
    rejected = envelope()
    rejected["canonical_state"] = dict(rejected["canonical_state"], observable_region_result_valid=False)
    payload = build_canonical_shadow_api_payload(monitor, [accepted, rejected])
    assert payload["mode"] == "read_only_shadow"
    assert payload["warning_actions_available"] is False
    assert payload["samples"][0]["measurements"] is not None
    assert payload["samples"][1]["measurements"] is None


def test_sidecar_database_round_trip_uses_separate_path(tmp_path: Path) -> None:
    canonical, downstream = summaries()
    monitor = build_shadow_monitor_summary(canonical, downstream)
    path = tmp_path / "shadow" / "audit.db"
    write_shadow_audit(path, "run", "protocol", monitor, [envelope()])
    result = read_shadow_audit(path, "run")
    assert result["run"]["authoritative"] == 0
    assert result["run"]["formal_warning_generated"] == 0
    assert result["samples"][0]["s8_status"] == "warning_suppressed"
    assert result["samples"][0]["eligible_for_downstream"] == 0


def test_monitoring_script_does_not_import_or_run_formal_agent() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "build_phase2d_c9_shadow_monitoring_snapshot.py").read_text(encoding="utf-8")
    assert "run_agent(" not in source
    assert "generate_warning_decision(" not in source
    assert '"http_server_started": False' in source
