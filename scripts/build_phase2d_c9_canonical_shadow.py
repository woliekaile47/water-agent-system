#!/usr/bin/env python3
"""Build canonical water-state shadow records from frozen C8 prediction outputs."""

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

from src.integration.canonical_water_state import (  # noqa: E402
    build_canonical_water_state,
    validate_canonical_water_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--candidate-gate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    geometry_root = args.geometry_root.expanduser().resolve()
    candidate_root = args.candidate_gate_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite canonical shadow output: {output_root}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["phase2d_c9_canonical_water_state"]
    geometry_summary_path = geometry_root / "geometry_stability_summary.json"
    candidate_summary_path = candidate_root / "candidate_gate_summary.json"
    geometry_summary = read_json(geometry_summary_path)
    candidate_summary = read_json(candidate_summary_path)
    geometry_rows = read_json(geometry_root / "per_frame_geometry_summary.json")
    candidate_rows = read_json(candidate_root / "candidate_gate_per_frame.json")
    if geometry_summary.get("ground_truth_used") is not False:
        raise ValueError("Geometry input is not prediction-only")
    if candidate_summary.get("ground_truth_used") is not False:
        raise ValueError("Candidate-gate input is not prediction-only")
    if len(geometry_rows) != len(candidate_rows):
        raise ValueError("Geometry and candidate-gate frame counts differ")

    candidates = {
        (str(row["sample_id"]), int(row["frame_index"])): row
        for row in candidate_rows
    }
    records: list[dict[str, Any]] = []
    for geometry in geometry_rows:
        key = (str(geometry["sample_id"]), int(geometry["frame_index"]))
        candidate = candidates.get(key)
        if candidate is None:
            raise ValueError(f"Missing candidate-gate decision for {key}")
        record = build_canonical_water_state(
            geometry,
            candidate,
            config,
            provenance={
                "geometry_dataset_sha256": sha256_file(geometry_summary_path),
                "candidate_gate_dataset_sha256": sha256_file(candidate_summary_path),
                "canonical_config_sha256": sha256_file(config_path),
                "source_mask_sha256": geometry.get("mask_sha256"),
            },
        )
        validate_canonical_water_state(record)
        records.append(record)

    latest_by_sample: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = record["identity"]["sample_id"]
        previous = latest_by_sample.get(sample_id)
        if previous is None or record["identity"]["frame_index"] > previous["identity"]["frame_index"]:
            latest_by_sample[sample_id] = record

    mismatch = Counter(
        (
            record["quality"]["legacy_runtime_gate"]["status"],
            record["quality"]["candidate_gate"]["status"],
        )
        for record in records
    )
    summary = {
        "protocol_version": "phase2d_c9a_canonical_water_state_shadow_v1",
        "schema_version": config["schema_version"],
        "deployment_mode": "shadow",
        "record_count": len(records),
        "sample_count": len(latest_by_sample),
        "legacy_candidate_status_matrix": {
            f"legacy_{legacy}__candidate_{candidate}": count
            for (legacy, candidate), count in sorted(mismatch.items())
        },
        "candidate_visible_status_counts": dict(sorted(Counter(
            record["quality"]["candidate_gate"]["status"] for record in records
        ).items())),
        "global_estimate_status_counts": dict(sorted(Counter(
            record["global_estimate_status"] for record in records
        ).items())),
        "downstream_eligible_count": sum(record["eligible_for_downstream"] for record in records),
        "authoritative_count": sum(record["authoritative"] for record in records),
        "ground_truth_used": False,
        "s5_s8_modified": False,
        "agent_modified": False,
        "dashboard_modified": False,
    }
    output_root.mkdir(parents=True)
    write_json(output_root / "canonical_water_state_records.json", records)
    write_json(output_root / "canonical_current_water_state_by_sample.json", latest_by_sample)
    write_json(output_root / "canonical_shadow_summary.json", summary)
    with (output_root / "canonical_water_state_records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
