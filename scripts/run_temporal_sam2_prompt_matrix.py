#!/usr/bin/env python3
"""Run the frozen Phase 2D-C-6B-2 GT-free automatic-prompt matrix."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path, key: str) -> dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    return document[key]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--matrix-config", type=Path, default=PROJECT_ROOT / "configs" / "phase2d_c6b2_heldout_matrix.yaml")
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "temporal_sam2_prompt.yaml",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite frozen matrix output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    matrix = _load_yaml(args.matrix_config.resolve(), "phase2d_c6b2_heldout_matrix")
    rows: list[dict[str, Any]] = []
    for sample in matrix["samples"]:
        sample_id = str(sample["sample_id"])
        sample_output = output_root / sample_id
        command = [
            sys.executable, str(root / "scripts" / "generate_temporal_sam2_prompt.py"),
            "--frames-dir", str(root / sample["frames_dir"]),
            "--image", str(root / sample["image"]),
            "--frame-index", str(sample["frame_index"]),
            "--expected-image-sha256", str(sample["image_sha256"]),
            "--config", str(args.prompt_config.resolve()),
            "--output-dir", str(sample_output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode not in (0, 2):
            raise RuntimeError(f"{sample_id} failed with {completed.returncode}: {completed.stderr}")
        prompt = _read_json(sample_output / "automatic_prompt.json")
        diagnostics = _read_json(sample_output / "automatic_prompt_diagnostics.json")
        rows.append({
            "sample_id": sample_id, "case_id": sample["case_id"], "rain_level": sample["rain_level"],
            "seed": int(sample["seed"]), "frame_index": int(sample["frame_index"]),
            "image_sha256": sample["image_sha256"], "prompt_status": prompt["prompt_quality_status"],
            "prompt_reasons": prompt["prompt_quality_reasons"], "component_count": diagnostics["component_count"],
            "positive_point_count": diagnostics["positive_point_count"],
            "negative_point_count": diagnostics["negative_point_count"],
            "negative_direction_sector_count": diagnostics["negative_direction_sector_count"],
            "prompt_path": str((sample_output / "automatic_prompt.json").resolve()),
            "reference_image": str((root / sample["image"]).resolve()),
            "sam2_started": False, "ground_truth_used": False, "eligible_for_downstream": False,
        })
    fieldnames = [
        "sample_id", "case_id", "rain_level", "seed", "frame_index", "image_sha256", "prompt_status",
        "prompt_reasons", "component_count", "positive_point_count", "negative_point_count",
        "negative_direction_sector_count", "prompt_path", "reference_image", "sam2_started",
        "ground_truth_used", "eligible_for_downstream",
    ]
    with (output_root / "prompt_matrix_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "prompt_reasons": ";".join(row["prompt_reasons"])})
    status_counts = {
        status: sum(row["prompt_status"] == status for row in rows)
        for status in ("pass", "diagnostic_only", "reject")
    }
    _write_json(output_root / "prompt_matrix_summary.json", {
        "protocol_version": matrix["protocol_version"], "selection_rule": matrix["selection_rule"],
        "sample_count": len(rows), "status_counts": status_counts, "sam2_started": False,
        "ground_truth_used": False, "eligible_for_downstream": False, "samples": rows,
    })
    print(json.dumps({"sample_count": len(rows), "status_counts": status_counts,
                      "output_root": str(output_root), "ground_truth_used": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
