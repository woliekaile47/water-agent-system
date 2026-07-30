"""Protocol tests for the C7-4 independent scalar evaluator."""

from pathlib import Path

from scripts.run_sam2_video_geometry_gt_evaluation import (
    resolve_evaluation_sequence_relative,
    save_error_chart,
)


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


def test_explicit_evaluation_sequence_path_is_separate_from_prediction_frames() -> None:
    sample = {
        "frames_dir": "outputs/staged/frames",
        "evaluation_sequence_dir": "data/simulation_dynamic/case/rain/seed",
    }
    assert resolve_evaluation_sequence_relative(sample) == sample["evaluation_sequence_dir"]


def test_missing_sequence_provenance_uses_verified_dataset_fallback() -> None:
    source = _source()
    assert 'dataset.get("ground_truth_used")' in source
    assert '"verified_dataset_summary_fallback"' in source


def test_empty_scalar_evaluation_writes_unavailable_chart(tmp_path: Path) -> None:
    output = tmp_path / "unavailable.png"
    save_error_chart(
        [{"frame_index": 0, "evaluation_available": False}],
        "unused_metric",
        output,
        "Unavailable evaluation",
    )
    assert output.is_file()
    assert output.stat().st_size > 0


def test_empty_scalar_evaluation_has_safe_overall_summary() -> None:
    source = _source()
    assert '"available_frame_count": sum(' in source
    assert 'item.get("water_level_within_3cm_count", 0)' in source
