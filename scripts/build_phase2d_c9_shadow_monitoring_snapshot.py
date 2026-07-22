#!/usr/bin/env python3
"""Build Agent/DB/API/Dashboard sidecar artifacts from C9 shadow outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.canonical_shadow_monitor import build_shadow_monitor_summary  # noqa: E402
from src.api.canonical_shadow_api import build_canonical_shadow_api_payload  # noqa: E402
from src.database.canonical_shadow_audit import read_shadow_audit, write_shadow_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_optional(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    config = yaml.safe_load(args.config.expanduser().resolve().read_text(encoding="utf-8"))[
        "phase2d_c9_shadow_monitoring"
    ]
    if config.get("mode") != "read_only_sidecar" or config.get("start_http_server") is not False:
        raise ValueError("C9-C permits only a read-only sidecar without an HTTP server")
    if config.get("modify_formal_agent") is not False or config.get("allow_warning_actions") is not False:
        raise ValueError("C9-C config attempted to modify formal Agent or warning actions")
    output_root = root / config["output_root"]
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite C9-C snapshot: {output_root}")

    canonical_root = root / config["canonical_root"]
    s5_s8_root = root / config["s5_s8_shadow_root"]
    canonical_summary = read_json(canonical_root / "canonical_shadow_summary.json")
    s5_s8_summary = read_json(s5_s8_root / "s5_s8_shadow_summary.json")
    envelopes = read_json(s5_s8_root / "s5_s8_shadow_envelopes.json")
    formal_db = root / config["formal_audit_db"]
    formal_db_hash_before = sha256_optional(formal_db)

    monitor = build_shadow_monitor_summary(canonical_summary, s5_s8_summary)
    if monitor["monitor_status"] != "healthy":
        raise RuntimeError("Shadow monitor safety checks failed")
    api_payload = build_canonical_shadow_api_payload(monitor, envelopes)
    output_root.mkdir(parents=True)
    sidecar_db = output_root / config["sidecar_audit_db_name"]
    write_shadow_audit(
        sidecar_db,
        run_id=config["run_id"],
        protocol_version=config["protocol_version"],
        monitor=monitor,
        envelopes=envelopes,
    )
    audit_readback = read_shadow_audit(sidecar_db, config["run_id"])
    formal_db_hash_after = sha256_optional(formal_db)
    if formal_db_hash_before != formal_db_hash_after:
        raise RuntimeError("Formal Agent audit database changed during sidecar monitoring")

    status = {
        "protocol_version": config["protocol_version"],
        "mode": "read_only_sidecar",
        "monitor": monitor,
        "sample_count": len(envelopes),
        "api_schema_version": api_payload["api_schema_version"],
        "sidecar_audit_db": str(sidecar_db),
        "sidecar_audit_row_count": len(audit_readback["samples"]),
        "formal_audit_db": str(formal_db),
        "formal_audit_db_sha256_before": formal_db_hash_before,
        "formal_audit_db_sha256_after": formal_db_hash_after,
        "formal_audit_db_unchanged": formal_db_hash_before == formal_db_hash_after,
        "http_server_started": False,
        "formal_agent_executed": False,
        "formal_warning_generated": False,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
    }
    write_json(output_root / "current_shadow_status.json", status)
    write_json(output_root / "canonical_shadow_api_payload.json", api_payload)
    write_json(output_root / "agent_shadow_monitor_summary.json", monitor)
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
