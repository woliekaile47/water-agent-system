"""Tests for Phase 2D-C-7-2 independent video-propagation GT evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.evaluation import evaluate_sam2_video_propagation_gt as evaluation


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    project = tmp_path / "project"
    propagation = tmp_path / "propagation"
    sample = {
        "sample_id": "c7_test",
        "case_id": "sim_water_10cm_001",
        "rain_level": "heavy",
        "seed": 302,
        "frames_dir": "frames",
    }
    frame_records = []
    rows = []
    output = propagation / sample["sample_id"]
    (project / "frames").mkdir(parents=True)
    (output / "masks_npy").mkdir(parents=True)
    for local, original in enumerate((9, 10, 11)):
        source = project / "frames" / f"frame_{original:06d}.png"
        source.write_bytes(f"rgb-{original}".encode())
        mask_path = output / "masks_npy" / f"frame_{original:06d}.npy"
        np.save(mask_path, np.eye(4, dtype=bool))
        frame_records.append({"original_frame_index": original, "source_sha256": _sha(source)})
        rows.append({
            "original_frame_index": original,
            "mask_sha256": _sha(mask_path),
            "is_anchor_frame": original == 10,
            "previous_frame_iou": None if local == 0 else 1.0,
        })
    summary = {
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
        "sam2_video_propagation_completed": True,
        "window_start": 9,
        "window_end": 11,
        "anchor_frame_index": 10,
        "source_frame_records": frame_records,
    }
    (output / "video_propagation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (output / "frame_metrics.json").write_text(json.dumps(rows), encoding="utf-8")
    return sample, project, propagation


def test_verification_checks_every_rgb_and_mask_before_gt(tmp_path: Path) -> None:
    sample, project, propagation = _frozen_fixture(tmp_path)
    result = evaluation.verify_frozen_video_sample(project, propagation, sample, 9, 11, 10)
    assert result["verified_before_ground_truth_read"] is True
    assert result["frame_count"] == 3
    assert [item["frame_index"] for item in result["frames"]] == [9, 10, 11]


def test_verification_fails_closed_on_mask_hash_change(tmp_path: Path) -> None:
    sample, project, propagation = _frozen_fixture(tmp_path)
    np.save(propagation / "c7_test/masks_npy/frame_000011.npy", np.zeros((4, 4), dtype=bool))
    try:
        evaluation.verify_frozen_video_sample(project, propagation, sample, 9, 11, 10)
    except ValueError as exc:
        assert "mask SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("Tampered frozen mask was accepted")


def test_frame_metrics_reuse_existing_camera_definitions() -> None:
    predicted = np.zeros((20, 20), dtype=bool)
    truth = np.zeros_like(predicted)
    predicted[4:14, 5:15] = True
    truth[4:14, 4:14] = True
    result = evaluation.evaluate_frozen_frame(predicted, truth)
    assert result["camera_mask_metrics"]["iou"] == 90 / 110
    assert result["camera_boundary_metrics"]["symmetric_outer_boundary"]["p95_px"] == 1.0


def test_sequence_summary_keeps_anchor_and_directional_groups_separate() -> None:
    rows = []
    for frame, iou in ((9, 0.8), (10, 0.9), (11, 0.7)):
        rows.append({
            "frame_index": frame,
            "iou": iou,
            "precision": iou,
            "recall": iou,
            "f1": iou,
            "outer_boundary_p95_px": 2.0,
            "predicted_pixels": 100,
            "previous_frame_iou": None if frame == 9 else 0.95,
            "area_absolute_error_pixels": 5,
        })
    summary = evaluation.summarize_sequence(rows, 10)
    assert summary["anchor"]["iou"] == 0.9
    assert summary["pre_anchor_iou"]["mean"] == 0.8
    assert summary["post_anchor_iou"]["mean"] == 0.7
    assert summary["worst_frame"]["frame_index"] == 11


def test_correlation_returns_none_for_constant_input() -> None:
    assert evaluation.pearson_correlation([1, 1, 1], [1, 2, 3]) is None


def test_evaluation_module_does_not_import_prediction_runner() -> None:
    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    assert "run_sam2_video_propagation" not in source
    assert "sam2.build_sam" not in source


def test_runner_uses_no_matplotlib_binary_dependency() -> None:
    runner = Path(__file__).resolve().parents[1] / "scripts/run_sam2_video_propagation_gt_evaluation.py"
    assert "matplotlib" not in runner.read_text(encoding="utf-8")


def test_runner_verifies_all_frozen_inputs_before_first_gt_read() -> None:
    runner = Path(__file__).resolve().parents[1] / "scripts/run_sam2_video_propagation_gt_evaluation.py"
    source = runner.read_text(encoding="utf-8")
    verification = source.index("frozen = {")
    first_gt_read = source.index("load_camera_mask_ground_truth(", verification)
    assert verification < first_gt_read
    assert "verify_frozen_video_sample" in source[verification:first_gt_read]
