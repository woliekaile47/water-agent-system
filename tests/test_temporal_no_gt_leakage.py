import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.perception.temporal_water_pipeline import run_temporal_prediction


def _config():
    return {
        "preprocessing": {"gaussian_blur_kernel": 1},
        "candidates": {"residual_threshold": 5, "residual_percentile": 99,
            "morphology_close_kernel": 1, "morphology_dilate_iterations": 0,
            "min_area_px": 2, "max_area_px": 200, "max_aspect_ratio": 10},
        "tracking": {"max_gap_frames": 1, "max_center_distance_px": 8,
            "min_bbox_iou_for_distant_match": 0.01},
        "classification": {"minimum_track_observations": 2, "water_score_threshold": .4,
            "dry_score_threshold": .4, "minimum_score_margin": .05},
        "evidence": {"minimum_kernel_sigma_px": 2, "maximum_propagation_radius_px": 8,
            "probability_scale": 1, "water_probability_threshold": .3,
            "unknown_evidence_threshold": .08, "morphology_close_kernel": 1},
        "ablation": {"shuffle_seed": 7},
    }


def test_prediction_interface_and_outputs_are_independent_of_corrupt_gt(tmp_path: Path):
    sequence = tmp_path / "sequence"
    frames = sequence / "frames"
    gt = sequence / "ground_truth"
    metadata = sequence / "metadata"
    frames.mkdir(parents=True)
    gt.mkdir()
    metadata.mkdir()
    for index in range(5):
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        image[10:13, 8 + index:11 + index] = 50 + index * 20
        Image.fromarray(image).save(frames / f"frame_{index:06d}.png")
    (sequence / "manifest.json").write_text("not valid json", encoding="utf-8")
    (gt / "event_annotations.json").write_text("corrupt", encoding="utf-8")
    (metadata / "event_states.json").write_text("corrupt", encoding="utf-8")
    first = run_temporal_prediction(str(frames), _config())
    (sequence / "manifest.json").write_text(json.dumps({"water_level": 999}), encoding="utf-8")
    (gt / "event_annotations.json").write_bytes(b"\x00\xffchanged")
    (metadata / "event_states.json").write_text(json.dumps({"water_event_count": 999}), encoding="utf-8")
    second = run_temporal_prediction(str(frames), _config())
    isolated_frames = tmp_path / "isolated" / "frames"
    isolated_frames.mkdir(parents=True)
    for source in sorted(frames.glob("frame_*.png")):
        Image.open(source).save(isolated_frames / source.name)
    isolated = run_temporal_prediction(str(isolated_frames), _config())
    assert list(inspect.signature(run_temporal_prediction).parameters) == ["frames_dir", "detector_config", "mode"]
    assert np.array_equal(first["evidence"]["predicted_water_mask"], second["evidence"]["predicted_water_mask"])
    assert np.array_equal(first["evidence"]["predicted_water_mask"], isolated["evidence"]["predicted_water_mask"])
    assert first["ground_truth_used"] is False
