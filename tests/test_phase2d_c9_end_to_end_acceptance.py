"""Tests for final C9-D invariants and fault injection."""

from pathlib import Path

import yaml

from src.evaluation.phase2d_c9_end_to_end_acceptance import (
    evaluate_end_to_end_invariants,
    hashes_unchanged,
    load_json_strict,
    run_fault_injection_suite,
)
from src.integration.canonical_water_state import build_canonical_water_state


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = yaml.safe_load(
    (ROOT / "configs" / "phase2d_c9_canonical_water_state.yaml").read_text(encoding="utf-8")
)["phase2d_c9_canonical_water_state"]


def canonical_state():
    geometry = {
        "sample_id": "sample", "frame_index": 169, "geometry_available": True,
        "quality_status": "reject", "gate_reasons": ["boundary"],
        "estimated_water_level_m": -0.2, "mean_depth_cm": 8.0,
        "median_depth_cm": 7.0, "max_depth_cm": 20.0,
        "water_area_m2": 7.0, "water_volume_m3": 0.6,
        "ground_truth_used": False,
    }
    candidate = {
        "sample_id": "sample", "case_id": "case", "rain_level": "moderate",
        "seed": 303, "frame_index": 169, "camera_visible_status": "pass",
        "global_scene_status": "complete", "visible_reject_reasons": [],
        "global_scope_reasons": [], "warnings": [], "ground_truth_used": False,
    }
    return build_canonical_water_state(
        geometry, candidate, CANONICAL_CONFIG, {"ground_truth_used": False}
    )


def summaries():
    canonical = {
        "record_count": 1, "sample_count": 1, "ground_truth_used": False,
        "downstream_eligible_count": 0, "authoritative_count": 0,
        "candidate_visible_status_counts": {"pass": 1, "reject": 0},
        "global_estimate_status_counts": {"complete": 1, "partial": 0},
        "legacy_candidate_status_matrix": {"legacy_reject__candidate_pass": 1},
    }
    downstream = {
        "ground_truth_used": False, "warning_generation_allowed_count": 0,
        "downstream_eligible_count": 0, "formal_output_hashes_unchanged": True,
        "formal_s5_s8_executed": False, "s5_status_counts": {"compatible_candidate": 1},
        "s7_status_counts": {"not_ready": 1}, "s8_status_counts": {"warning_suppressed": 1},
    }
    return canonical, downstream


def envelope():
    return {
        "canonical_state": canonical_state(),
        "s5_shadow_input": {"status": "compatible_candidate"},
        "s7_shadow_preflight": {"status": "not_ready"},
        "s8_shadow_decision": {"status": "warning_suppressed"},
        "authoritative": False, "eligible_for_downstream": False,
    }


def test_all_defined_faults_are_rejected(tmp_path: Path) -> None:
    canonical, downstream = summaries()
    results = run_fault_injection_suite(
        tmp_path, canonical_state(), canonical, downstream, envelope()
    )
    assert len(results) == 9
    assert all(item["status"] == "pass" for item in results)
    assert all(item["fault_rejected"] for item in results)


def test_invariants_accept_a_safe_shadow_chain() -> None:
    canonical, downstream = summaries()
    state = canonical_state()
    monitoring = {
        "monitor": {"monitor_status": "healthy"},
        "formal_audit_db_unchanged": True,
        "http_server_started": False,
        "formal_agent_executed": False,
        "formal_warning_generated": False,
    }
    api = {
        "mode": "read_only_shadow", "warning_actions_available": False,
        "samples": [{"candidate_status": "pass", "measurements": state["measurements"]}],
    }
    checks = evaluate_end_to_end_invariants(
        canonical, [state], downstream, [envelope()], monitoring, api, 1,
        {"canonical_record_count": 1, "sample_count": 1, "candidate_pass_count": 1,
         "candidate_reject_count": 0, "global_partial_count": 0},
    )
    assert all(item["passed"] for item in checks)


def test_rejected_api_measurement_exposure_fails_invariant() -> None:
    canonical, downstream = summaries()
    state = canonical_state()
    monitoring = {
        "monitor": {"monitor_status": "healthy"}, "formal_audit_db_unchanged": True,
        "http_server_started": False, "formal_agent_executed": False,
        "formal_warning_generated": False,
    }
    api = {
        "mode": "read_only_shadow", "warning_actions_available": False,
        "samples": [{"candidate_status": "reject", "measurements": {"mean_depth_cm": 8.0}}],
    }
    checks = evaluate_end_to_end_invariants(
        canonical, [state], downstream, [envelope()], monitoring, api, 1,
        {"canonical_record_count": 1, "sample_count": 1, "candidate_pass_count": 1,
         "candidate_reject_count": 0, "global_partial_count": 0},
    )
    result = next(item for item in checks if item["check"] == "api_rejected_measurements_hidden")
    assert result["passed"] is False


def test_hash_comparison_detects_mutation() -> None:
    assert hashes_unchanged({"a": "1"}, {"a": "1"}) is True
    assert hashes_unchanged({"a": "1"}, {"a": "2"}) is False


def test_strict_json_loader_rejects_missing_file(tmp_path: Path) -> None:
    try:
        load_json_strict(tmp_path / "missing.json")
    except FileNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("missing JSON was accepted")


def test_acceptance_script_never_starts_runtime_services() -> None:
    source = (ROOT / "scripts" / "run_phase2d_c9_end_to_end_acceptance.py").read_text(encoding="utf-8")
    assert "run_agent(" not in source
    assert "uvicorn" not in source
    assert "subprocess" not in source
    assert "os.system" not in source
    assert '"manual_demo_required": True' in source
    assert '"http_server_started": False' in source
