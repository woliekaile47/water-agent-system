"""Protocol tests for the C8 seed-303 automatic-prompt runner."""

from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_phase2d_c8_seed303_prompt_matrix.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_all_anchor_hashes_are_verified_before_first_prompt_subprocess() -> None:
    source = _source()
    verification = source.index("# Verify every anchor before generating the first prompt.")
    generation = source.index("completed = subprocess.run(")
    assert verification < generation


def test_runner_does_not_read_ground_truth_or_start_sam2() -> None:
    source = _source()
    for forbidden in (
        "camera_water_mask_gt", "dem_water_mask_gt", "depth_map_gt",
        "water_level_gt", "build_sam2", "run_video_frame_geometry",
    ):
        assert forbidden not in source
    assert '"sam2_started": False' in source
    assert '"ground_truth_used": False' in source


def test_runner_refuses_to_overwrite_frozen_output() -> None:
    source = _source()
    assert "if output_root.exists() and not args.resume_partial:" in source
    assert "refusing to overwrite frozen prompt output" in source


def test_resume_reuses_complete_prompts_and_rejects_partial_directories() -> None:
    source = _source()
    assert "reused_existing = sample_output.exists()" in source
    assert "partial existing prompt cannot be resumed" in source
    assert 'prompt.get("ground_truth_used") is not False' in source


def test_parallelism_is_bounded() -> None:
    source = _source()
    assert "if args.jobs < 1 or args.jobs > 3:" in source
    assert "ThreadPoolExecutor(max_workers=args.jobs)" in source
