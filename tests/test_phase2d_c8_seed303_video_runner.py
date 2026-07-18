"""Protocol tests for the C8 seed-303 SAM2 video matrix runner."""

from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_phase2d_c8_seed303_video_matrix.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_all_inputs_are_verified_before_first_sam2_subprocess() -> None:
    source = _source()
    verification = source.index("# Verify all RGB windows and prompts before the first SAM2 process")
    subprocess_call = source.index("completed = subprocess.run(")
    assert verification < subprocess_call


def test_runner_refuses_overwrite_and_rejected_prompts() -> None:
    source = _source()
    assert "if output.exists():" in source
    assert "refusing to overwrite frozen video matrix" in source
    assert 'prompt.get("prompt_quality_status") == "reject"' in source


def test_runner_has_no_ground_truth_or_geometry_dependency() -> None:
    source = _source()
    for forbidden in (
        "camera_water_mask_gt", "dem_water_mask_gt", "depth_map_gt", "water_level_gt",
        "run_video_frame_geometry", "candidate_gate",
    ):
        assert forbidden not in source
    assert '"ground_truth_used": False' in source
    assert '"sam2_rerun_count": 0' in source
