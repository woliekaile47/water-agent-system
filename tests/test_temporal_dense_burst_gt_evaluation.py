"""Protocol tests for dense-burst temporal-mask GT evaluation."""

from pathlib import Path


def _source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluate_temporal_dense_burst_gt.py"
    ).read_text(encoding="utf-8")


def test_frozen_artifacts_are_verified_before_gt_read() -> None:
    source = _source()
    verification = source.index("frozen = verify_frozen_prediction(args)")
    gt_read = source.index("load_camera_mask_ground_truth(", verification)
    assert verification < gt_read


def test_evaluator_does_not_run_prediction_or_sam2() -> None:
    source = _source()
    for forbidden in (
        "run_temporal_prediction",
        "generate_temporal_sam2_prompt",
        "build_sam2",
        "run_sam2_video_propagation",
    ):
        assert forbidden not in source
    assert '"sam2_run_count": 0' in source
    assert '"ground_truth_used_for_prediction": False' in source
