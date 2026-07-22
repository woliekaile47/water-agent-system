#!/usr/bin/env python3
"""Run final C9 synthetic-shadow acceptance and temporary fault injection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.phase2d_c9_end_to_end_acceptance import (  # noqa: E402
    evaluate_end_to_end_invariants,
    hashes_unchanged,
    load_json_strict,
    run_fault_injection_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def sha256_optional(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    config = yaml.safe_load(args.config.expanduser().resolve().read_text(encoding="utf-8"))[
        "phase2d_c9_end_to_end_acceptance"
    ]
    policy = config["acceptance_policy"]
    if policy.get("allow_formal_warning") is not False or policy.get("allow_downstream") is not False:
        raise ValueError("C9-D acceptance cannot enable warnings or downstream use")
    output_root = root / config["output_root"]
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite C9-D acceptance output: {output_root}")

    protected = [root / path for path in config["protected_formal_paths"]]
    protected_before = {str(path): sha256_optional(path) for path in protected}
    canonical_root = root / config["canonical_root"]
    s5_s8_root = root / config["s5_s8_shadow_root"]
    monitoring_root = root / config["monitoring_root"]
    canonical_summary = load_json_strict(canonical_root / "canonical_shadow_summary.json")
    canonical_records = load_json_strict(canonical_root / "canonical_water_state_records.json")
    s5_s8_summary = load_json_strict(s5_s8_root / "s5_s8_shadow_summary.json")
    envelopes = load_json_strict(s5_s8_root / "s5_s8_shadow_envelopes.json")
    monitoring_status = load_json_strict(monitoring_root / "current_shadow_status.json")
    api_payload = load_json_strict(monitoring_root / "canonical_shadow_api_payload.json")
    audit_path = monitoring_root / "canonical_shadow_audit.db"
    with sqlite3.connect(audit_path) as connection:
        audit_sample_count = int(connection.execute("SELECT count(1) FROM shadow_sample_states").fetchone()[0])
        unsafe_audit_count = int(connection.execute(
            "SELECT count(1) FROM shadow_sample_states WHERE authoritative != 0 OR eligible_for_downstream != 0 OR s8_status != 'warning_suppressed'"
        ).fetchone()[0])

    invariants = evaluate_end_to_end_invariants(
        canonical_summary, canonical_records, s5_s8_summary, envelopes,
        monitoring_status, api_payload, audit_sample_count, config["expected"],
    )
    invariants.append({
        "check": "sidecar_audit_all_safe",
        "status": "pass" if unsafe_audit_count == 0 else "fail",
        "passed": unsafe_audit_count == 0,
    })
    dashboard_source = (root / "dashboard" / "app.py").read_text(encoding="utf-8")
    dashboard_ok = "C9 Shadow：统一状态监控" in dashboard_source and "warning actions unavailable" in dashboard_source
    invariants.append({"check": "dashboard_shadow_page_present", "status": "pass" if dashboard_ok else "fail", "passed": dashboard_ok})

    with tempfile.TemporaryDirectory(prefix="phase2d_c9d_faults_") as directory:
        faults = run_fault_injection_suite(
            Path(directory), canonical_records[0], canonical_summary,
            s5_s8_summary, envelopes[0],
        )
    protected_after = {str(path): sha256_optional(path) for path in protected}
    protected_ok = hashes_unchanged(protected_before, protected_after)
    invariants.append({
        "check": "all_protected_formal_files_unchanged",
        "status": "pass" if protected_ok else "fail",
        "passed": protected_ok,
    })

    invariant_pass = all(item["passed"] for item in invariants)
    fault_pass = all(item["status"] == "pass" and item["fault_rejected"] for item in faults)
    accepted = invariant_pass and fault_pass
    summary = {
        "protocol_version": config["protocol_version"],
        "acceptance_status": "pass" if accepted else "fail",
        "competition_demo_readiness": "ready_for_synthetic_shadow_demo" if accepted else "not_ready",
        "production_readiness": False,
        "real_world_validated": False,
        "formal_warning_activation_allowed": False,
        "invariant_count": len(invariants),
        "invariant_pass_count": sum(item["passed"] for item in invariants),
        "fault_scenario_count": len(faults),
        "fault_rejected_count": sum(item["fault_rejected"] and item["status"] == "pass" for item in faults),
        "protected_formal_hashes_before": protected_before,
        "protected_formal_hashes_after": protected_after,
        "protected_formal_files_unchanged": protected_ok,
        "http_server_started": False,
        "dashboard_started": False,
        "formal_agent_executed": False,
        "formal_warning_generated": False,
        "ground_truth_used_for_prediction": False,
        "authoritative": False,
        "eligible_for_downstream": False,
    }
    output_root.mkdir(parents=True)
    write_json(output_root / "acceptance_summary.json", summary)
    write_json(output_root / "invariant_results.json", invariants)
    write_json(output_root / "fault_injection_results.json", faults)
    write_json(output_root / "demo_readiness.json", {
        "status": summary["competition_demo_readiness"],
        "scope": policy["competition_demo_scope"],
        "manual_dashboard_command": "streamlit run dashboard/app.py",
        "manual_demo_required": True,
        "formal_warning_activation_allowed": False,
        "real_world_claim_allowed": False,
    })
    report = [
        "# Phase 2D-C-9D 端到端验收",
        "",
        f"- Acceptance: {summary['acceptance_status']}",
        f"- Invariants: {summary['invariant_pass_count']}/{summary['invariant_count']}",
        f"- Faults rejected: {summary['fault_rejected_count']}/{summary['fault_scenario_count']}",
        f"- Competition demo: {summary['competition_demo_readiness']}",
        "- Production / real-world validation: false",
        "- Formal warning activation: false",
        "- All faults were injected only into temporary copies.",
        "",
    ]
    (output_root / "acceptance_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
