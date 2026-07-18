#!/usr/bin/env python3
"""Independent Camera-mask GT evaluation for frozen temporal-prompt SAM 2 masks.

This module is evaluation-only.  It deliberately loads only Camera water-mask
Ground Truth and never imports or calls a prediction runner.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.evaluation.evaluate_sam2_shoreline_geometry_gt import (
    binary_mask_metrics,
    boundary_metrics,
    enclosed_hole_metrics,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_camera_mask_ground_truth(
    project_root: str | Path,
    case_id: str,
    rain_level: str,
    seed: int,
    frame_index: int,
) -> dict[str, Any]:
    """Load and validate only strict same-state Camera mask GT for evaluation."""
    root = Path(project_root).expanduser().resolve()
    case_dir = root / "data" / "simulation" / case_id
    sequence_dir = (
        root
        / "data"
        / "simulation_dynamic"
        / case_id
        / rain_level
        / f"seed_{int(seed)}"
    )
    paths = {
        "case_manifest": case_dir / "manifest.json",
        "sequence_manifest": sequence_dir / "metadata" / "sequence_manifest.json",
        "case_camera_mask": case_dir / "ground_truth" / "camera_water_mask_gt.png",
        "sequence_camera_mask": sequence_dir / "ground_truth" / "water_mask.png",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Camera-mask evaluation inputs: {missing}")

    case_manifest = json.loads(paths["case_manifest"].read_text(encoding="utf-8"))
    sequence_manifest = json.loads(paths["sequence_manifest"].read_text(encoding="utf-8"))
    if case_manifest.get("case_id") != case_id or sequence_manifest.get("case_id") != case_id:
        raise ValueError("Case and sequence manifests do not identify the requested case")
    if sequence_manifest.get("rain_level") != rain_level:
        raise ValueError("Sequence manifest rain level differs from the requested sample")
    if int(sequence_manifest.get("random_seed", -1)) != int(seed):
        raise ValueError("Sequence manifest seed differs from the requested sample")
    frame_count = int(sequence_manifest.get("frame_count", 0))
    if not 0 <= int(frame_index) < frame_count:
        raise ValueError("Frame index is outside the sequence frame range")

    case_raw = np.asarray(Image.open(paths["case_camera_mask"]).convert("L"))
    sequence_raw = np.asarray(Image.open(paths["sequence_camera_mask"]).convert("L"))
    for label, raw in (("case", case_raw), ("sequence", sequence_raw)):
        if raw.shape != (360, 640):
            raise ValueError(f"{label} Camera GT shape is not 360x640")
        if not set(np.unique(raw).tolist()).issubset({0, 255}):
            raise ValueError(f"{label} Camera GT is not binary 0/255")
    case_mask = case_raw > 127
    sequence_mask = sequence_raw > 127
    if not np.array_equal(case_mask, sequence_mask):
        raise ValueError("Case and sequence Camera GT masks are not a strict static-state match")

    return {
        "camera_mask": case_mask,
        "validation": {
            "data_role": "independent_camera_mask_ground_truth_evaluation",
            "case_id": case_id,
            "rain_level": rain_level,
            "seed": int(seed),
            "frame_index": int(frame_index),
            "frame_count": frame_count,
            "camera_mask_shape_hw": list(case_mask.shape),
            "camera_water_pixel_count": int(np.count_nonzero(case_mask)),
            "camera_mask_binary_0_255": True,
            "case_sequence_camera_mask_equal": True,
            "static_water_state_applies_to_requested_frame": True,
            "gt_fields_loaded": ["camera_water_mask"],
            "gt_fields_not_loaded": [
                "water_level",
                "dem_water_mask",
                "depth_map",
                "area",
                "volume",
            ],
            "paths": {key: str(path) for key, path in paths.items()},
        },
    }


def verify_frozen_sample_inputs(
    sample: dict[str, Any],
    prompt_sample: dict[str, Any],
    sam2_output_root: str | Path,
) -> dict[str, Any]:
    """Verify frozen RGB and SAM 2 mask hashes before any GT is opened."""
    if sample["sample_id"] != prompt_sample["sample_id"]:
        raise ValueError("Prompt and SAM 2 summaries have different sample IDs")
    for key in ("case_id", "rain_level", "seed", "frame_index"):
        if sample[key] != prompt_sample[key]:
            raise ValueError(f"Prompt and SAM 2 metadata differ for {key}")
    image_path = Path(prompt_sample["reference_image"]).expanduser().resolve()
    mask_path = Path(sam2_output_root).expanduser().resolve() / sample["sample_id"] / "prompted_mask_raw.npy"
    summary_path = Path(sam2_output_root).expanduser().resolve() / sample["sample_id"] / "prompted_mask_summary.json"
    for path in (image_path, mask_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_image_hash = sha256_file(image_path)
    actual_mask_hash = sha256_file(mask_path)
    actual_summary_hash = sha256_file(summary_path)
    if actual_image_hash != prompt_sample["image_sha256"]:
        raise ValueError(f"Frozen RGB hash mismatch for {sample['sample_id']}")
    if actual_mask_hash != sample["mask_sha256"]:
        raise ValueError(f"Frozen mask hash mismatch for {sample['sample_id']}")
    if actual_summary_hash != sample["summary_sha256"]:
        raise ValueError(f"Frozen SAM 2 summary hash mismatch for {sample['sample_id']}")
    return {
        "verified": True,
        "verified_before_ground_truth_read": True,
        "image_path": str(image_path),
        "image_sha256": actual_image_hash,
        "mask_path": str(mask_path),
        "mask_sha256": actual_mask_hash,
        "sam2_summary_path": str(summary_path),
        "sam2_summary_sha256": actual_summary_hash,
    }


def evaluate_camera_mask(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Evaluate a frozen binary mask without changing either input."""
    prediction = np.asarray(predicted, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    before_prediction = prediction.copy()
    before_target = target.copy()
    result = {
        "camera_mask_metrics": binary_mask_metrics(prediction, target),
        "camera_boundary_metrics": boundary_metrics(prediction, target),
        "prediction_topology": enclosed_hole_metrics(prediction),
    }
    if not np.array_equal(prediction, before_prediction) or not np.array_equal(target, before_target):
        raise RuntimeError("Evaluation mutated an input mask")
    camera = result["camera_mask_metrics"]
    outer = result["camera_boundary_metrics"]["symmetric_outer_boundary"]
    result["offline_research_criteria"] = {
        "source": "previously_declared_Phase_2D_C_single_frame_research_readiness",
        "not_a_prediction_side_quality_gate": True,
        "thresholds": {"iou_min": 0.90, "recall_min": 0.90, "outer_boundary_p95_px_max": 5.0},
        "iou_met": bool(camera["iou"] >= 0.90),
        "recall_met": bool(camera["recall"] >= 0.90),
        "outer_boundary_p95_met": bool(
            outer["p95_px"] is not None and outer["p95_px"] <= 5.0
        ),
    }
    result["offline_research_criteria"]["all_met"] = bool(
        result["offline_research_criteria"]["iou_met"]
        and result["offline_research_criteria"]["recall_met"]
        and result["offline_research_criteria"]["outer_boundary_p95_met"]
    )
    return result
