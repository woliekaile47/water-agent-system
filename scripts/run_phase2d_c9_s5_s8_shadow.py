#!/usr/bin/env python3
"""Run S5-S8 contract checks from canonical states without formal pipeline writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.canonical_s5_s8_shadow import build_s5_s8_shadow_envelope  # noqa: E402


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
    root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    canonical_root = args.canonical_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite C9-B shadow output: {output_root}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["phase2d_c9_s5_s8_shadow"]
    if config.get("deployment_mode") != "shadow":
        raise ValueError("C9-B only supports shadow mode")
    forbidden = ("execute_s5", "execute_s6", "execute_s7", "execute_s8", "allow_formal_output_writes", "allow_warning_generation")
    if any(config.get(key) is not False for key in forbidden):
        raise ValueError("C9-B config attempted to enable a formal stage or warning write")

    formal_paths = [root / path for path in config["formal_output_paths"]]
    hashes_before = {str(path): sha256_optional(path) for path in formal_paths}
    records = read_json(canonical_root / "canonical_water_state_records.json")
    canonical_summary = read_json(canonical_root / "canonical_shadow_summary.json")
    if canonical_summary.get("ground_truth_used") is not False:
        raise ValueError("Canonical shadow input has invalid GT provenance")
    if any(record.get("eligible_for_downstream") is not False for record in records):
        raise ValueError("Canonical input contains downstream-eligible records")

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sample[record["identity"]["sample_id"]].append(record)
    envelopes: list[dict[str, Any]] = []
    for sample_id, states in sorted(by_sample.items()):
        ordered = sorted(states, key=lambda state: int(state["identity"]["frame_index"]))
        envelope = build_s5_s8_shadow_envelope(
            latest_state=ordered[-1],
            sequence_states=ordered,
            fps=float(config["fps"]),
            minimum_history_minutes=float(config["s7_minimum_history_minutes"]),
        )
        envelopes.append(envelope)

    hashes_after = {str(path): sha256_optional(path) for path in formal_paths}
    unchanged = hashes_before == hashes_after
    if not unchanged:
        raise RuntimeError("A formal S5-S8 output changed during shadow execution")
    summary = {
        "protocol_version": config["protocol_version"],
        "deployment_mode": "shadow",
        "sample_count": len(envelopes),
        "s5_status_counts": dict(sorted(Counter(
            item["s5_shadow_input"]["status"] for item in envelopes
        ).items())),
        "s7_status_counts": dict(sorted(Counter(
            item["s7_shadow_preflight"]["status"] for item in envelopes
        ).items())),
        "s8_status_counts": dict(sorted(Counter(
            item["s8_shadow_decision"]["status"] for item in envelopes
        ).items())),
        "warning_generation_allowed_count": sum(
            item["s8_shadow_decision"]["warning_generation_allowed"] for item in envelopes
        ),
        "downstream_eligible_count": sum(item["eligible_for_downstream"] for item in envelopes),
        "formal_output_hashes_before": hashes_before,
        "formal_output_hashes_after": hashes_after,
        "formal_output_hashes_unchanged": unchanged,
        "formal_s5_s8_executed": False,
        "formal_warning_output_written": False,
        "ground_truth_used": False,
        "agent_modified": False,
        "dashboard_modified": False,
    }
    output_root.mkdir(parents=True)
    write_json(output_root / "s5_s8_shadow_envelopes.json", envelopes)
    write_json(output_root / "s5_s8_shadow_summary.json", summary)
    with (output_root / "s5_s8_shadow_envelopes.jsonl").open("w", encoding="utf-8") as stream:
        for envelope in envelopes:
            stream.write(json.dumps(envelope, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
