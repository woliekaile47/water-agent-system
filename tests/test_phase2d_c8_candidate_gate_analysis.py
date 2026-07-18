"""Protocol tests for the C8-2 offline comparison script."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_phase2d_c8_candidate_gate.py"
)


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_candidate_decisions_are_completed_before_evaluation_files_are_opened() -> None:
    source = _source()
    candidate_call = source.index("candidate = evaluate_candidate_gate(")
    evaluation_read = source.index("# GT is opened only after every prediction-side candidate decision")
    assert candidate_call < evaluation_read


def test_script_does_not_run_prediction_or_sam2() -> None:
    source = _source()
    for forbidden in ("build_sam2", "run_video_frame_geometry", "run_sam2_video_geometry_stability"):
        assert forbidden not in source
    assert '"prediction_recomputed": False' in source
    assert '"sam2_rerun_count": 0' in source


def test_candidate_output_remains_non_authoritative_and_ineligible() -> None:
    source = _source()
    assert '"authoritative": False' in source
    assert '"eligible_for_downstream": False' in source
