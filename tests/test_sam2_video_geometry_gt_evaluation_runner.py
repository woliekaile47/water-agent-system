"""Protocol tests for the C7-4 independent scalar evaluator."""

from pathlib import Path


def _source() -> str:
    return (
        Path(__file__).resolve().parents[1] / "scripts/run_sam2_video_geometry_gt_evaluation.py"
    ).read_text(encoding="utf-8")


def test_all_frozen_prediction_files_are_verified_before_gt_loader_call() -> None:
    source = _source()
    verification = source.index("frozen = verify_frozen_geometry_outputs(")
    gt_read = source.index("load_ground_truth_evaluation_inputs(", verification)
    assert verification < gt_read


def test_evaluator_does_not_call_prediction_or_sam2() -> None:
    source = _source()
    for forbidden in (
        "run_video_frame_geometry",
        "run_sam2_video_geometry_stability",
        "build_sam2",
        "sam2_video_predictor",
    ):
        assert forbidden not in source
    assert '"geometry_prediction_recomputed_count": 0' in source
    assert '"ground_truth_used_for_prediction": False' in source


def test_per_cell_metrics_are_explicitly_unavailable() -> None:
    source = _source()
    assert '"per_cell_metrics_available": False' in source
    assert "per-frame prediction rasters were not frozen by C7-3" in source
