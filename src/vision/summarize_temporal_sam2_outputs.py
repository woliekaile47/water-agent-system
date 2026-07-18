#!/usr/bin/env python3
"""Summarize frozen SAM 2 prompt outputs without Ground Truth evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_frozen_sam2_outputs(
    prompt_matrix_summary: str | Path,
    sam2_output_root: str | Path,
) -> dict[str, Any]:
    """Validate and summarize frozen candidate masks; never read GT artifacts."""
    prompt_summary_path = Path(prompt_matrix_summary).expanduser().resolve()
    output_root = Path(sam2_output_root).expanduser().resolve()
    prompt_matrix = _read_json(prompt_summary_path)
    rows: list[dict[str, Any]] = []
    for prompt_row in prompt_matrix["samples"]:
        sample_id = str(prompt_row["sample_id"])
        sample_dir = output_root / sample_id
        summary_path = sample_dir / "prompted_mask_summary.json"
        mask_path = sample_dir / "prompted_mask_raw.npy"
        if not summary_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"missing frozen SAM2 output for {sample_id}")
        summary = _read_json(summary_path)
        if summary.get("semantic_label") != "unknown_candidate" or summary.get("authoritative") is not False:
            raise ValueError(f"invalid candidate semantics for {sample_id}")
        if summary.get("prompt_source") != "temporal_water_evidence_v1":
            raise ValueError(f"unexpected prompt source for {sample_id}")
        candidates = list(summary.get("candidates", []))
        if not candidates:
            raise ValueError(f"no SAM2 candidates returned for {sample_id}")
        selected_id = int(summary["selected_candidate_id"])
        selected = next((item for item in candidates if int(item["candidate_id"]) == selected_id), None)
        if selected is None:
            raise ValueError(f"selected candidate is absent for {sample_id}")
        maximum_score = max(float(item["score"]) for item in candidates)
        selected_score = float(summary["selected_score"])
        selected_is_highest = abs(selected_score - maximum_score) <= 1e-9
        if not selected_is_highest:
            raise ValueError(f"selected candidate is not highest-score for {sample_id}")
        mask = np.asarray(np.load(mask_path), dtype=bool)
        if mask.ndim != 2 or mask.shape != (360, 640):
            raise ValueError(f"unexpected mask shape for {sample_id}: {mask.shape}")
        mask_pixels = int(np.count_nonzero(mask))
        if mask_pixels != int(summary["raw_mask_area_pixels"]):
            raise ValueError(f"raw mask area mismatch for {sample_id}")
        rows.append({
            "sample_id": sample_id,
            "case_id": prompt_row["case_id"],
            "rain_level": prompt_row["rain_level"],
            "seed": int(prompt_row["seed"]),
            "frame_index": int(prompt_row["frame_index"]),
            "prompt_status": prompt_row["prompt_status"],
            "returned_candidate_count": int(summary["returned_candidate_count"]),
            "selected_candidate_id": selected_id,
            "selected_score": selected_score,
            "selected_candidate_is_highest_score": selected_is_highest,
            "raw_mask_area_pixels": mask_pixels,
            "raw_mask_area_ratio": float(summary["raw_mask_area_ratio"]),
            "connected_component_count": int(summary["connected_component_count"]),
            "enclosed_hole_count": int(summary["enclosed_hole_count"]),
            "inference_time_seconds": float(summary["inference_time_seconds"]),
            "total_time_seconds": float(summary["total_time_seconds"]),
            "peak_gpu_allocated_mib": float(summary["peak_gpu_allocated_mib"]),
            "peak_gpu_reserved_mib": float(summary["peak_gpu_reserved_mib"]),
            "cuda_oom": bool(summary["cuda_oom"]),
            "mask_sha256": _sha256(mask_path),
            "summary_sha256": _sha256(summary_path),
            "semantic_label": "unknown_candidate",
            "authoritative": False,
            "ground_truth_used": False,
            "eligible_for_downstream": False,
        })
    if len(rows) != int(prompt_matrix["sample_count"]):
        raise ValueError("SAM2 output count does not match frozen prompt matrix")
    return {
        "protocol_version": "phase2d_c6b2_sam2_output_summary_v1",
        "sample_count": len(rows),
        "prompt_status_counts": prompt_matrix["status_counts"],
        "sam2_run_count": len(rows),
        "rerun_count": 0,
        "cuda_oom_count": sum(row["cuda_oom"] for row in rows),
        "selected_highest_score_count": sum(row["selected_candidate_is_highest_score"] for row in rows),
        "mask_area_pixels": {
            "minimum": min(row["raw_mask_area_pixels"] for row in rows),
            "median": float(np.median([row["raw_mask_area_pixels"] for row in rows])),
            "maximum": max(row["raw_mask_area_pixels"] for row in rows),
        },
        "selected_score": {
            "minimum": min(row["selected_score"] for row in rows),
            "median": float(np.median([row["selected_score"] for row in rows])),
            "maximum": max(row["selected_score"] for row in rows),
        },
        "peak_gpu_allocated_mib_max": max(row["peak_gpu_allocated_mib"] for row in rows),
        "automatic_prompt_not_water_semantic_prediction": True,
        "runner_static_manual_guidance_note_is_not_authoritative_metadata": True,
        "ground_truth_used": False,
        "eligible_for_downstream": False,
        "samples": rows,
    }


def write_sam2_output_summary(result: dict[str, Any], output_root: str | Path) -> None:
    root = Path(output_root).expanduser().resolve()
    _write_json(root / "sam2_output_matrix_summary.json", result)
    rows = result["samples"]
    with (root / "sam2_output_matrix_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
