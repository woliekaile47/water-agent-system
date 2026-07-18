#!/usr/bin/env python3
"""Evaluation-only helpers for frozen SAM 2 video propagation masks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.evaluate_temporal_sam2_mask_gt import evaluate_camera_mask


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_video_sample(
    project_root: str | Path,
    propagation_root: str | Path,
    sample: dict[str, Any],
    window_start: int,
    window_end: int,
    anchor_frame_index: int,
) -> dict[str, Any]:
    """Verify every RGB and mask hash without opening any Ground Truth."""
    root = Path(project_root).expanduser().resolve()
    output_dir = Path(propagation_root).expanduser().resolve() / sample["sample_id"]
    summary_path = output_dir / "video_propagation_summary.json"
    metrics_path = output_dir / "frame_metrics.json"
    if not summary_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Missing frozen propagation metadata in {output_dir}")
    summary = _read_json(summary_path)
    rows = _read_json(metrics_path)
    expected_indices = list(range(int(window_start), int(window_end) + 1))
    if summary.get("ground_truth_used") is not False:
        raise ValueError("Frozen propagation provenance is not GT-free")
    if summary.get("authoritative") is not False or summary.get("eligible_for_downstream") is not False:
        raise ValueError("Frozen propagation output has unsafe result semantics")
    if summary.get("sam2_video_propagation_completed") is not True:
        raise ValueError("Frozen propagation did not complete")
    if [summary.get("window_start"), summary.get("window_end"), summary.get("anchor_frame_index")] != [
        int(window_start), int(window_end), int(anchor_frame_index)
    ]:
        raise ValueError("Frozen propagation window differs from the evaluation matrix")
    if len(rows) != len(expected_indices) or len(summary.get("source_frame_records", [])) != len(expected_indices):
        raise ValueError("Frozen propagation frame count differs from the evaluation matrix")

    frames_dir = root / sample["frames_dir"]
    verified_frames: list[dict[str, Any]] = []
    for expected, row, source in zip(expected_indices, rows, summary["source_frame_records"]):
        if int(row["original_frame_index"]) != expected or int(source["original_frame_index"]) != expected:
            raise ValueError("Frozen propagation frame ordering is inconsistent")
        if bool(row["is_anchor_frame"]) != (expected == int(anchor_frame_index)):
            raise ValueError("Frozen anchor-frame marker is inconsistent")
        source_path = frames_dir / f"frame_{expected:06d}.png"
        mask_path = output_dir / "masks_npy" / f"frame_{expected:06d}.npy"
        if not source_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing frozen RGB or mask for frame {expected}")
        source_hash = sha256_file(source_path)
        mask_hash = sha256_file(mask_path)
        if source_hash != source["source_sha256"]:
            raise ValueError(f"Frozen RGB SHA-256 mismatch at frame {expected}")
        if mask_hash != row["mask_sha256"]:
            raise ValueError(f"Frozen mask SHA-256 mismatch at frame {expected}")
        verified_frames.append({
            "frame_index": expected,
            "source_path": str(source_path),
            "source_sha256": source_hash,
            "mask_path": str(mask_path),
            "mask_sha256": mask_hash,
            "previous_frame_iou": row.get("previous_frame_iou"),
            "is_anchor_frame": bool(row["is_anchor_frame"]),
        })
    return {
        "sample_id": sample["sample_id"],
        "verified": True,
        "verified_before_ground_truth_read": True,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "frame_metrics_path": str(metrics_path),
        "frame_metrics_sha256": sha256_file(metrics_path),
        "frame_count": len(verified_frames),
        "frames": verified_frames,
    }


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[finite]
    y_array = y_array[finite]
    if x_array.size < 2 or np.std(x_array) == 0 or np.std(y_array) == 0:
        return None
    return float(np.corrcoef(x_array, y_array)[0, 1])


def summarize_sequence(rows: list[dict[str, Any]], anchor_frame_index: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty sequence")
    by_frame = {int(row["frame_index"]): row for row in rows}
    if anchor_frame_index not in by_frame:
        raise ValueError("Anchor frame is absent from evaluated rows")
    anchor = by_frame[anchor_frame_index]
    before = [row for row in rows if int(row["frame_index"]) < anchor_frame_index]
    after = [row for row in rows if int(row["frame_index"]) > anchor_frame_index]
    gt_iou = [float(row["iou"]) for row in rows]
    distances = [float(abs(int(row["frame_index"]) - anchor_frame_index)) for row in rows]
    adjacent_rows = [row for row in rows if row["previous_frame_iou"] is not None]
    worst = min(rows, key=lambda row: float(row["iou"]))
    return {
        "frame_count": len(rows),
        "iou": metric_summary(gt_iou),
        "precision": metric_summary([float(row["precision"]) for row in rows]),
        "recall": metric_summary([float(row["recall"]) for row in rows]),
        "f1": metric_summary([float(row["f1"]) for row in rows]),
        "outer_boundary_p95_px": metric_summary([float(row["outer_boundary_p95_px"]) for row in rows]),
        "predicted_area_pixels": metric_summary([float(row["predicted_pixels"]) for row in rows]),
        "anchor": {
            "frame_index": anchor_frame_index,
            "iou": float(anchor["iou"]),
            "outer_boundary_p95_px": float(anchor["outer_boundary_p95_px"]),
            "predicted_pixels": int(anchor["predicted_pixels"]),
        },
        "pre_anchor_iou": metric_summary([float(row["iou"]) for row in before]),
        "post_anchor_iou": metric_summary([float(row["iou"]) for row in after]),
        "endpoint_iou": {
            "first": float(rows[0]["iou"]),
            "last": float(rows[-1]["iou"]),
        },
        "worst_frame": {
            "frame_index": int(worst["frame_index"]),
            "iou": float(worst["iou"]),
            "outer_boundary_p95_px": float(worst["outer_boundary_p95_px"]),
        },
        "correlations": {
            "distance_from_anchor_vs_gt_iou": pearson_correlation(distances, gt_iou),
            "adjacent_iou_vs_gt_iou": pearson_correlation(
                [float(row["previous_frame_iou"]) for row in adjacent_rows],
                [float(row["iou"]) for row in adjacent_rows],
            ),
            "absolute_area_error_vs_gt_iou": pearson_correlation(
                [float(row["area_absolute_error_pixels"]) for row in rows], gt_iou
            ),
        },
    }


def evaluate_frozen_frame(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Reuse the established Camera-mask evaluation formulas unchanged."""
    return evaluate_camera_mask(predicted, truth)
