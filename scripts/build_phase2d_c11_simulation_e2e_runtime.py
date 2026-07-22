#!/usr/bin/env python3
"""Build C11-A simulation-only routing envelopes from frozen C9 canonical states."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.simulation_e2e_runtime import (  # noqa: E402
    build_simulation_runtime_envelope,
    validate_simulation_runtime_config,
    validate_simulation_runtime_envelope,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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
    project_root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    canonical_root = args.canonical_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite C11-A output: {output_root}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["phase2d_c11_simulation_e2e_runtime"]
    validate_simulation_runtime_config(config)
    records = read_json(canonical_root / "canonical_water_state_records.json")
    canonical_summary = read_json(canonical_root / "canonical_shadow_summary.json")
    if canonical_summary.get("ground_truth_used") is not False:
        raise ValueError("Canonical source summary has invalid GT provenance")

    formal_paths = [project_root / item for item in config["formal_output_paths"]]
    hashes_before = {str(path): sha256_optional(path) for path in formal_paths}
    envelopes = [build_simulation_runtime_envelope(record, config) for record in records]
    for envelope in envelopes:
        validate_simulation_runtime_envelope(envelope)

    latest_by_sample: dict[str, dict[str, Any]] = {}
    for envelope in envelopes:
        sample_id = envelope["identity"]["sample_id"]
        previous = latest_by_sample.get(sample_id)
        if previous is None or envelope["identity"]["frame_index"] > previous["identity"]["frame_index"]:
            latest_by_sample[sample_id] = envelope

    hashes_after = {str(path): sha256_optional(path) for path in formal_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("A formal S5-S8 output changed during C11-A routing")

    routing_counts = Counter(item["simulation_routing"]["status"] for item in envelopes)
    summary = {
        "protocol_version": config["protocol_version"],
        "runtime_mode": "simulation_e2e",
        "data_domain": "simulation",
        "record_count": len(envelopes),
        "sample_count": len(latest_by_sample),
        "routing_status_counts": dict(sorted(routing_counts.items())),
        "simulation_pipeline_eligible_count": sum(
            item["simulation_routing"]["eligible_for_simulation_pipeline"] for item in envelopes
        ),
        "simulation_global_s7_s8_eligible_count": sum(
            item["simulation_routing"]["eligible_for_simulation_global_s7_s8"] for item in envelopes
        ),
        "real_warning_eligible_count": sum(item["safety"]["eligible_for_real_warning"] for item in envelopes),
        "external_notification_allowed_count": sum(
            item["safety"]["external_notification_allowed"] for item in envelopes
        ),
        "formal_output_hashes_before": hashes_before,
        "formal_output_hashes_after": hashes_after,
        "formal_output_hashes_unchanged": True,
        "ground_truth_used": False,
        "formal_s5_s8_executed": False,
        "real_warning_generated": False,
        "agent_modified": False,
        "dashboard_modified": False,
    }
    output_root.mkdir(parents=True)
    write_json(output_root / "simulation_runtime_envelopes.json", envelopes)
    write_json(output_root / "simulation_runtime_current_by_sample.json", latest_by_sample)
    write_json(output_root / "simulation_runtime_summary.json", summary)
    with (output_root / "simulation_runtime_envelopes.jsonl").open("w", encoding="utf-8") as stream:
        for envelope in envelopes:
            stream.write(json.dumps(envelope, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
