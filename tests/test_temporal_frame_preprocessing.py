from pathlib import Path

import numpy as np
from PIL import Image

from src.perception.temporal_frame_preprocessing import load_detector_frames, preprocess_temporal_frames


def test_frames_only_loader_and_exposure_correction(tmp_path: Path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index, value in enumerate((20, 40, 60)):
        image = np.full((12, 16, 3), value, dtype=np.uint8)
        image[5:7, 7:9] = min(255, value + index * 20)
        Image.fromarray(image).save(frames_dir / f"frame_{index:06d}.png")
    frames, loader = load_detector_frames(frames_dir)
    result, diagnostics = preprocess_temporal_frames(frames, {"gaussian_blur_kernel": 1})
    assert frames.shape == (3, 12, 16, 3)
    assert loader["ground_truth_or_metadata_read"] is False
    assert result["absolute_residual"].shape == (3, 12, 16)
    assert diagnostics["uses_water_mask"] is False


def test_loader_rejects_non_frames_directory(tmp_path: Path):
    directory = tmp_path / "images"
    directory.mkdir()
    try:
        load_detector_frames(directory)
    except ValueError as error:
        assert "frames directory" in str(error)
    else:
        raise AssertionError("non-frames detector input must be rejected")
