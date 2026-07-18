#!/usr/bin/env python3
"""Apply the frozen C8 candidate gate to frozen seed-303 geometry outputs."""

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

from src.evaluation.phase2d_c8_candidate_quality_gate import evaluate_candidate_gate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--matrix-config", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--geometry-config-key", default="phase2d_c8_seed303_geometry_confirmation")
    parser.add_argument("--candidate-gate-config", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
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


def summarize_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    visible = Counter(item["camera_visible_status"] for item in decisions)
    global_scene = Counter(item["global_scene_status"] for item in decisions)
    warnings = Counter(warning for item in decisions for warning in item["warnings"])
    reasons = Counter(reason for item in decisions for reason in item["visible_reject_reasons"])
    return {
        "frame_count": len(decisions),
        "camera_visible_status_counts": dict(sorted(visible.items())),
        "global_scene_status_counts": dict(sorted(global_scene.items())),
        "visible_reject_reason_counts": dict(sorted(reasons.items())),
        "warning_counts": dict(sorted(warnings.items())),
        "boundary_only_reject_count": sum(bool(item["boundary_metric_rejected_by_itself"]) for item in decisions),
    }


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    matrix_path = args.matrix_config.expanduser().resolve()
    geometry_config_path = args.geometry_config.expanduser().resolve()
    gate_path = args.candidate_gate_config.expanduser().resolve()
    geometry_root = args.geometry_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite candidate-gate freeze: {output_root}")

    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))["phase2d_c8_seed303_confirmation_matrix"]
    geometry_config = yaml.safe_load(geometry_config_path.read_text(encoding="utf-8"))[args.geometry_config_key]
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))["phase2d_c8_candidate_quality_gate"]
    expected_gate_sha = matrix["frozen_candidate_gate_config"]["sha256"]
    if sha256_file(gate_path) != expected_gate_sha:
        raise ValueError("candidate gate changed after seed-303 confirmation matrix freeze")
    if gate.get("replaces_runtime_gate") is not False or gate.get("eligible_for_downstream") is not False:
        raise ValueError("candidate gate is not an offline, non-downstream research gate")

    dataset_path = geometry_root / "geometry_stability_summary.json"
    dataset = read_json(dataset_path)
    expected_total = int(matrix["sample_count"]) * int(matrix["window_frame_count"])
    if dataset.get("frame_count") != expected_total or dataset.get("ground_truth_used") is not False:
        raise ValueError("geometry freeze provenance or frame count mismatch")
    if dataset.get("gate_thresholds_modified") is not False or dataset.get("sam2_rerun_count") != 0:
        raise ValueError("geometry freeze changed thresholds or reran SAM2")

    output_root.mkdir(parents=True)
    all_decisions: list[dict[str, Any]] = []
    per_sequence: dict[str, Any] = {}
    for sample in geometry_config["samples"]:
        sample_id = sample["sample_id"]
        sample_dir = geometry_root / sample_id
        rows_path = sample_dir / "per_frame_geometry_summary.json"
        sequence_path = sample_dir / "sequence_geometry_stability.json"
        rows = read_json(rows_path)
        sequence = read_json(sequence_path)
        expected_frames = list(range(int(geometry_config["window_start"]), int(geometry_config["window_end"]) + 1))
        if [int(row["frame_index"]) for row in rows] != expected_frames:
            raise ValueError(f"geometry frame order mismatch for {sample_id}")
        decisions = []
        for row in rows:
            decision = evaluate_candidate_gate(row, sequence, gate)
            decision.update({
                "sample_id": sample_id,
                "case_id": sample["case_id"],
                "rain_level": sample["rain_level"],
                "seed": int(sample["seed"]),
                "geometry_row_sha256_source": sha256_file(rows_path),
            })
            decisions.append(decision)
            all_decisions.append(decision)
        sequence_output = output_root / sample_id
        sequence_output.mkdir()
        write_json(sequence_output / "candidate_gate_per_frame.json", decisions)
        sequence_summary = summarize_decisions(decisions)
        sequence_summary.update({
            "sample_id": sample_id,
            "case_id": sample["case_id"],
            "rain_level": sample["rain_level"],
            "seed": int(sample["seed"]),
            "geometry_rows_sha256": sha256_file(rows_path),
            "geometry_sequence_sha256": sha256_file(sequence_path),
            "ground_truth_used": False,
            "authoritative": False,
            "eligible_for_downstream": False,
        })
        write_json(sequence_output / "candidate_gate_sequence_summary.json", sequence_summary)
        per_sequence[sample_id] = sequence_summary

    summary = summarize_decisions(all_decisions)
    summary.update({
        "protocol_version": "phase2d_c8_seed303_candidate_gate_freeze_v1",
        "sample_count": len(geometry_config["samples"]),
        "matrix_config_sha256": sha256_file(matrix_path),
        "geometry_config_sha256": sha256_file(geometry_config_path),
        "geometry_dataset_sha256": sha256_file(dataset_path),
        "candidate_gate_config_sha256": sha256_file(gate_path),
        "candidate_gate_frozen_before_confirmation": True,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
        "sequences": per_sequence,
    })
    write_json(output_root / "candidate_gate_summary.json", summary)
    write_json(output_root / "candidate_gate_per_frame.json", all_decisions)
    (output_root / "run_log.txt").write_text(
        json.dumps({"status": "completed", "frame_count": len(all_decisions), "ground_truth_used": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
