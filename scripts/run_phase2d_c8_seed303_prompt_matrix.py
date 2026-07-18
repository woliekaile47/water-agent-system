#!/usr/bin/env python3
"""Generate frozen GT-free automatic prompts for the C8 seed-303 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=Path("configs/phase2d_c8_seed303_confirmation_matrix.yaml"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    matrix_path = args.matrix_config if args.matrix_config.is_absolute() else root / args.matrix_config
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))[
        "phase2d_c8_seed303_confirmation_matrix"
    ]
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.resume_partial:
        raise FileExistsError(f"refusing to overwrite frozen prompt output: {output_root}")
    if (output_root / "prompt_matrix_summary.json").exists():
        raise FileExistsError(f"prompt matrix is already frozen: {output_root}")
    if args.jobs < 1 or args.jobs > 3:
        raise ValueError("jobs must be between 1 and 3")

    prompt_config = root / matrix["automatic_prompt_config"]["path"]
    if sha256_file(prompt_config) != matrix["automatic_prompt_config"]["sha256"]:
        raise ValueError("automatic prompt config differs from frozen SHA-256")

    # Verify every anchor before generating the first prompt.
    for sample in matrix["samples"]:
        anchor = root / sample["anchor_image"]
        if sha256_file(anchor) != sample["anchor_sha256"]:
            raise ValueError(f"frozen anchor SHA-256 mismatch: {sample['sample_id']}")

    output_root.mkdir(parents=True, exist_ok=args.resume_partial)

    def generate_or_load(sample: dict[str, Any]) -> dict[str, Any]:
        sample_output = output_root / sample["sample_id"]
        prompt_path = sample_output / "automatic_prompt.json"
        diagnostics_path = sample_output / "automatic_prompt_diagnostics.json"
        preview_path = sample_output / "automatic_prompt_preview.png"
        reused_existing = sample_output.exists()
        if reused_existing:
            if not all(path.is_file() for path in (prompt_path, diagnostics_path, preview_path)):
                raise RuntimeError(f"partial existing prompt cannot be resumed: {sample['sample_id']}")
        else:
            command = [
                sys.executable,
                str(root / "scripts" / "generate_temporal_sam2_prompt.py"),
                "--frames-dir", str(root / sample["frames_dir"]),
                "--image", str(root / sample["anchor_image"]),
                "--frame-index", str(matrix["anchor_frame_index"]),
                "--expected-image-sha256", sample["anchor_sha256"],
                "--config", str(prompt_config),
                "--output-dir", str(sample_output),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode not in (0, 2):
                raise RuntimeError(
                    f"{sample['sample_id']} prompt generation failed: {completed.stderr}"
                )
        prompt = read_json(prompt_path)
        diagnostics = read_json(diagnostics_path)
        if prompt.get("image_sha256") != sample["anchor_sha256"]:
            raise ValueError(f"existing prompt anchor mismatch: {sample['sample_id']}")
        if prompt.get("ground_truth_used") is not False:
            raise ValueError(f"existing prompt is not GT-free: {sample['sample_id']}")
        return {
            "sample_id": sample["sample_id"],
            "case_id": sample["case_id"],
            "rain_level": sample["rain_level"],
            "seed": matrix["seed"],
            "frame_index": matrix["anchor_frame_index"],
            "anchor_sha256": sample["anchor_sha256"],
            "prompt_sha256": sha256_file(prompt_path),
            "prompt_status": prompt["prompt_quality_status"],
            "prompt_reasons": list(prompt["prompt_quality_reasons"]),
            "positive_point_count": diagnostics["positive_point_count"],
            "negative_point_count": diagnostics["negative_point_count"],
            "component_count": diagnostics["component_count"],
            "prompt_path": str(prompt_path.resolve()),
            "reused_existing_prompt": reused_existing,
            "sam2_started": False,
            "ground_truth_used": False,
            "eligible_for_downstream": False,
        }

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        generated = list(executor.map(generate_or_load, matrix["samples"]))
    rows = sorted(generated, key=lambda row: row["sample_id"])

    fields = list(rows[0])
    with (output_root / "prompt_matrix_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "prompt_reasons": "|".join(row["prompt_reasons"]),
            })
    status_counts = {
        status: sum(row["prompt_status"] == status for row in rows)
        for status in ("pass", "diagnostic_only", "reject")
    }
    write_json(output_root / "prompt_matrix_summary.json", {
        "protocol_version": "phase2d_c8_seed303_prompt_freeze_v1",
        "matrix_id": matrix["matrix_id"],
        "sample_count": len(rows),
        "all_anchor_hashes_verified_before_first_prompt": True,
        "status_counts": status_counts,
        "sam2_started": False,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
        "samples": rows,
    })
    print(json.dumps({"sample_count": len(rows), "status_counts": status_counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
