#!/usr/bin/env python3
"""RGB-only detector input loading and temporal residual preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def load_detector_frames(frames_dir: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load only PNG files from a frames directory; no parent metadata access."""
    directory = Path(frames_dir).expanduser().resolve()
    if not directory.is_dir() or directory.name != "frames":
        raise ValueError(f"detector_input_loader requires a frames directory: {directory}")
    paths = sorted(directory.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.png files in {directory}")
    frames = []
    expected_size = None
    for expected_index, path in enumerate(paths):
        if path.name != f"frame_{expected_index:06d}.png":
            raise ValueError(f"Non-contiguous detector frame numbering at {path.name}")
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if expected_size is None:
            expected_size = image.shape
        if image.shape != expected_size:
            raise ValueError(f"Inconsistent frame size: {image.shape} vs {expected_size}")
        frames.append(image)
    array = np.stack(frames)
    return array, {
        "input_role": "detector_rgb_frames_only",
        "frames_directory": str(directory),
        "frame_count": int(array.shape[0]),
        "height": int(array.shape[1]),
        "width": int(array.shape[2]),
        "ground_truth_or_metadata_read": False,
    }


def _color_mean_normalize(frames: np.ndarray) -> np.ndarray:
    values = frames.astype(np.float32)
    means = values.mean(axis=(1, 2), keepdims=True)
    reference = np.median(means, axis=0, keepdims=True)
    return np.clip(values - means + reference, 0.0, 255.0)


def preprocess_temporal_frames(
    frames_rgb: np.ndarray,
    config: dict[str, Any],
    mode: str = "full",
    shuffle_seed: int = 2027,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frames = np.asarray(frames_rgb, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames_rgb must have shape TxHxWx3")
    if mode == "single_frame":
        frames = frames[:1]
    elif mode == "shuffled":
        order = np.random.default_rng(int(shuffle_seed)).permutation(frames.shape[0])
        frames = frames[order]
    elif mode == "color_normalized":
        frames = _color_mean_normalize(frames).astype(np.uint8)
    elif mode != "full":
        raise ValueError(f"Unsupported preprocessing mode: {mode}")
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]).astype(np.float32)
    kernel = int(config.get("gaussian_blur_kernel", 3))
    if kernel > 1:
        gray = np.stack([cv2.GaussianBlur(frame, (kernel, kernel), 0) for frame in gray])
    frame_medians = np.median(gray, axis=(1, 2))
    reference = float(np.median(frame_medians))
    exposure_offsets = frame_medians - reference
    corrected = gray - exposure_offsets[:, None, None]
    signed = np.zeros_like(corrected, dtype=np.float32)
    if corrected.shape[0] > 1:
        signed[1:] = corrected[1:] - corrected[:-1]
    absolute = np.abs(signed)
    positive = np.maximum(signed, 0.0)
    negative = np.maximum(-signed, 0.0)
    preview = np.max(absolute, axis=0) if absolute.size else np.zeros(gray.shape[1:], dtype=np.float32)
    diagnostics = {
        "mode": mode,
        "frame_count": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "global_exposure_offset_median": float(np.median(exposure_offsets)),
        "global_exposure_offset_mad": float(np.median(np.abs(exposure_offsets - np.median(exposure_offsets)))),
        "exposure_anomaly_fraction": float(np.mean(np.abs(exposure_offsets) > 15.0)),
        "residual_mean": float(np.mean(absolute)),
        "residual_p99": float(np.percentile(absolute, 99.0)),
        "uses_water_mask": False,
    }
    return {
        "corrected_gray": corrected,
        "signed_residual": signed,
        "absolute_residual": absolute,
        "positive_residual": positive,
        "negative_residual": negative,
        "temporal_residual_preview": preview,
        "exposure_offsets": exposure_offsets.astype(np.float32),
    }, diagnostics
