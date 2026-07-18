"""Tests for Phase 2D-C-6B-3 independent Camera-mask GT evaluation."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.evaluation import evaluate_temporal_sam2_mask_gt as evaluation
from src.vision import temporal_sam2_prompt_pipeline as prediction


def _write_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_camera_mask_metrics_reuse_existing_exact_definitions() -> None:
    predicted = np.zeros((20, 20), dtype=bool)
    truth = np.zeros_like(predicted)
    predicted[4:14, 5:15] = True
    truth[4:14, 4:14] = True
    before = predicted.copy()
    first = evaluation.evaluate_camera_mask(predicted, truth)
    second = evaluation.evaluate_camera_mask(predicted, truth)
    assert first == second
    assert first["camera_mask_metrics"]["iou"] == 90 / 110
    assert first["camera_boundary_metrics"]["symmetric_outer_boundary"]["p95_px"] == 1.0
    assert np.array_equal(predicted, before)


def test_camera_only_gt_loader_validates_static_same_frame_without_other_gt(tmp_path: Path) -> None:
    root = tmp_path
    case = "sim_water_10cm_001"
    sequence = root / "data/simulation_dynamic" / case / "heavy/seed_301"
    case_dir = root / "data/simulation" / case
    case_dir.mkdir(parents=True)
    (case_dir / "manifest.json").write_text(json.dumps({"case_id": case}), encoding="utf-8")
    (sequence / "metadata").mkdir(parents=True)
    (sequence / "metadata/sequence_manifest.json").write_text(json.dumps({
        "case_id": case, "rain_level": "heavy", "random_seed": 301, "frame_count": 200,
    }), encoding="utf-8")
    mask = np.zeros((360, 640), dtype=bool)
    mask[100:130, 200:250] = True
    _write_png(case_dir / "ground_truth/camera_water_mask_gt.png", mask)
    _write_png(sequence / "ground_truth/water_mask.png", mask)
    loaded = evaluation.load_camera_mask_ground_truth(root, case, "heavy", 301, 149)
    assert np.array_equal(loaded["camera_mask"], mask)
    assert loaded["validation"]["gt_fields_loaded"] == ["camera_water_mask"]
    assert "depth_map" in loaded["validation"]["gt_fields_not_loaded"]


def test_camera_gt_loader_rejects_mismatch_and_nonbinary(tmp_path: Path) -> None:
    case = "sim_water_5cm_001"
    case_dir = tmp_path / "data/simulation" / case
    sequence = tmp_path / "data/simulation_dynamic" / case / "light/seed_301"
    case_dir.mkdir(parents=True)
    (case_dir / "manifest.json").write_text(json.dumps({"case_id": case}), encoding="utf-8")
    (sequence / "metadata").mkdir(parents=True)
    (sequence / "metadata/sequence_manifest.json").write_text(json.dumps({
        "case_id": case, "rain_level": "light", "random_seed": 301, "frame_count": 200,
    }), encoding="utf-8")
    base = np.zeros((360, 640), dtype=bool)
    other = base.copy()
    other[0, 0] = True
    _write_png(case_dir / "ground_truth/camera_water_mask_gt.png", base)
    _write_png(sequence / "ground_truth/water_mask.png", other)
    with pytest.raises(ValueError, match="strict static-state"):
        evaluation.load_camera_mask_ground_truth(tmp_path, case, "light", 301, 149)


def test_all_frozen_hashes_are_checked_before_gt_by_runner_source() -> None:
    runner = (
        Path(__file__).resolve().parents[1] / "scripts/run_temporal_sam2_mask_gt_evaluation.py"
    ).read_text(encoding="utf-8")
    verify_position = runner.index("frozen_checks =")
    gt_position = runner.index("load_camera_mask_ground_truth(", verify_position)
    assert verify_position < gt_position
    assert "verify_frozen_sample_inputs" in runner[verify_position:gt_position]


def test_frozen_hash_verification_fails_closed(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    mask = tmp_path / "sam2/sample/prompted_mask_raw.npy"
    summary = tmp_path / "sam2/sample/prompted_mask_summary.json"
    _write_png(image, np.zeros((4, 4), dtype=bool))
    mask.parent.mkdir(parents=True)
    np.save(mask, np.zeros((4, 4), dtype=bool))
    summary.write_text("{}\n", encoding="utf-8")
    sample = {
        "sample_id": "sample", "case_id": "case", "rain_level": "heavy", "seed": 301,
        "frame_index": 149, "mask_sha256": _sha(mask), "summary_sha256": _sha(summary),
    }
    prompt = {
        "sample_id": "sample", "case_id": "case", "rain_level": "heavy", "seed": 301,
        "frame_index": 149, "reference_image": str(image), "image_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="RGB hash mismatch"):
        evaluation.verify_frozen_sample_inputs(sample, prompt, tmp_path / "sam2")


def test_prediction_module_has_no_gt_dependency() -> None:
    source = inspect.getsource(prediction)
    for token in ("camera_water_mask_gt", "load_camera_mask_ground_truth", "evaluation_matrix_summary"):
        assert token not in source
