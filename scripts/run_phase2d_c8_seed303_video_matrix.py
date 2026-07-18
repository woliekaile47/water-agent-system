#!/usr/bin/env python3
"""Run SAM2 video propagation for the frozen C8 seed-303 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_sha256(frames_dir: Path, start: int, end: int) -> str:
    digest = hashlib.sha256()
    for index in range(start, end + 1):
        digest.update(bytes.fromhex(sha256_file(frames_dir / f"frame_{index:06d}.png")))
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange-root", required=True, type=Path)
    parser.add_argument("--propagation-script", required=True, type=Path)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exchange = args.exchange_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen video matrix: {output}")
    matrix = yaml.safe_load(
        (exchange / "phase2d_c8_seed303_confirmation_matrix.yaml").read_text(encoding="utf-8")
    )["phase2d_c8_seed303_confirmation_matrix"]
    prompt_summary = read_json(exchange / "prompt_matrix_summary.json")
    prompt_hashes = {
        item["sample_id"]: item["prompt_sha256"] for item in prompt_summary["samples"]
    }

    # Verify all RGB windows and prompts before the first SAM2 process starts.
    verified: list[dict[str, Any]] = []
    for sample in matrix["samples"]:
        sample_id = sample["sample_id"]
        frames = exchange / "frames" / sample_id
        prompt_path = exchange / "prompts" / sample_id / "automatic_prompt.json"
        prompt = read_json(prompt_path)
        if window_sha256(frames, matrix["window_start"], matrix["window_end"]) != sample[
            "window_sha256"
        ]:
            raise ValueError(f"frozen RGB window mismatch: {sample_id}")
        if sha256_file(frames / f"frame_{matrix['anchor_frame_index']:06d}.png") != sample[
            "anchor_sha256"
        ]:
            raise ValueError(f"frozen anchor mismatch: {sample_id}")
        if sha256_file(prompt_path) != prompt_hashes[sample_id]:
            raise ValueError(f"frozen prompt mismatch: {sample_id}")
        if prompt.get("ground_truth_used") is not False:
            raise ValueError(f"prompt is not GT-free: {sample_id}")
        if prompt.get("prompt_quality_status") == "reject":
            raise ValueError(f"rejected prompt cannot start SAM2: {sample_id}")
        verified.append({
            "sample_id": sample_id,
            "window_sha256": sample["window_sha256"],
            "anchor_sha256": sample["anchor_sha256"],
            "prompt_sha256": prompt_hashes[sample_id],
            "prompt_status": prompt["prompt_quality_status"],
        })

    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for sample in matrix["samples"]:
        sample_id = sample["sample_id"]
        sample_output = output / sample_id
        command = [
            sys.executable,
            str(args.propagation_script.resolve()),
            "--frames-dir", str((exchange / "frames" / sample_id).resolve()),
            "--prompt-config", str(
                (exchange / "prompts" / sample_id / "automatic_prompt.json").resolve()
            ),
            "--window-start", str(matrix["window_start"]),
            "--window-end", str(matrix["window_end"]),
            "--anchor-frame-index", str(matrix["anchor_frame_index"]),
            "--model-config", args.model_config,
            "--checkpoint", str(args.checkpoint.resolve()),
            "--output-dir", str(sample_output),
            "--device", args.device,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{sample_id} SAM2 propagation failed with {completed.returncode}: "
                f"{completed.stderr}"
            )
        summary_path = sample_output / "video_propagation_summary.json"
        summary = read_json(summary_path)
        rows.append({
            "sample_id": sample_id,
            "case_id": sample["case_id"],
            "rain_level": sample["rain_level"],
            "frame_count": summary["frame_count"],
            "prompt_status": summary["prompt_quality_status"],
            "area_cv": summary["mask_area_pixels"]["coefficient_of_variation"],
            "adjacent_mask_iou_minimum": summary["adjacent_mask_iou"]["minimum"],
            "adjacent_mask_iou_median": summary["adjacent_mask_iou"]["median"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "cuda_peak_allocated_mib": summary["cuda_peak_allocated_mib"],
            "summary_sha256": sha256_file(summary_path),
            "ground_truth_used": False,
            "authoritative": False,
            "eligible_for_downstream": False,
        })

    matrix_summary = {
        "protocol_version": "phase2d_c8_seed303_video_freeze_v1",
        "sample_count": len(rows),
        "frame_count": sum(row["frame_count"] for row in rows),
        "all_inputs_verified_before_first_sam2_process": True,
        "verified_inputs": verified,
        "sam2_run_count": len(rows),
        "sam2_rerun_count": 0,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
        "samples": rows,
    }
    (output / "video_matrix_summary.json").write_text(
        json.dumps(matrix_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "sample_count": len(rows),
        "frame_count": matrix_summary["frame_count"],
        "sam2_rerun_count": 0,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
