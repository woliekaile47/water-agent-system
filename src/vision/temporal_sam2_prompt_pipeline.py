#!/usr/bin/env python3
"""GT-free continuous-RGB to temporal SAM 2 prompt orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.perception.temporal_water_pipeline import run_temporal_prediction
from src.perception.temporal_water_quality_gate import evaluate_temporal_quality_gate
from src.vision.generate_temporal_sam2_prompt import (
    generate_temporal_sam2_prompt,
    sha256_file,
)


def _order_sensitivity(full: dict[str, Any], shuffled: dict[str, Any]) -> float:
    left = np.asarray(full["evidence"]["predicted_water_probability"], dtype=np.float64)
    right = np.asarray(shuffled["evidence"]["predicted_water_probability"], dtype=np.float64)
    difference = float(np.mean(np.abs(left - right)))
    reference = max(float(np.mean(left) + np.mean(right)), 0.01)
    return float(np.clip(difference / reference, 0.0, 1.0))


def run_temporal_sam2_prompt_from_frames(
    frames_dir: str | Path,
    image_path: str | Path,
    frame_index: int,
    detector_config: dict[str, Any],
    temporal_gate_config: dict[str, Any],
    prompt_config: dict[str, Any],
    *,
    expected_image_sha256: str | None = None,
) -> dict[str, Any]:
    """Run prediction-only temporal evidence and generate a frozen SAM 2 prompt."""
    frames = Path(frames_dir).expanduser().resolve()
    image = Path(image_path).expanduser().resolve()
    if not frames.is_dir() or frames.name != "frames":
        raise ValueError("frames_dir must be an existing directory named frames")
    if image.parent != frames:
        raise ValueError("reference image must be a direct child of frames_dir")
    expected_name = f"frame_{int(frame_index):06d}.png"
    if image.name != expected_name:
        raise ValueError(f"reference image name must be {expected_name}")
    image_sha256 = sha256_file(image)
    if expected_image_sha256 is not None and image_sha256 != expected_image_sha256.lower():
        raise ValueError("reference image SHA-256 does not match the frozen manifest")

    full = run_temporal_prediction(str(frames), detector_config, "full")
    shuffled = run_temporal_prediction(str(frames), detector_config, "shuffled")
    order_sensitivity = _order_sensitivity(full, shuffled)
    gate = evaluate_temporal_quality_gate(
        full["loader"],
        full["preprocessing_diagnostics"],
        full["candidate_diagnostics"],
        full["tracks"],
        full["classifications"],
        full["evidence"],
        full["evidence_diagnostics"],
        order_sensitivity,
        full["water_mask_time_stability"],
        full["feature_score_separation"],
        float(detector_config["fps"]),
        temporal_gate_config,
    )
    prompt, prompt_diagnostics = generate_temporal_sam2_prompt(
        full["evidence"]["predicted_water_probability"],
        full["evidence"]["predicted_water_mask"],
        full["evidence"]["predicted_unknown_mask"],
        full["classifications"],
        gate,
        prompt_config,
        image_path=str(image),
        image_sha256=image_sha256,
        frame_index=int(frame_index),
    )
    return {
        "prediction": full,
        "order_sensitivity": order_sensitivity,
        "temporal_quality_gate": gate,
        "prompt": prompt,
        "prompt_diagnostics": prompt_diagnostics,
        "reference_image_sha256_verified": expected_image_sha256 is not None,
        "ground_truth_used": False,
    }
