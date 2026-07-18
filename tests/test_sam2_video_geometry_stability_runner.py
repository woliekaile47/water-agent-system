"""Protocol tests for the C7-3 video geometry runner."""

from pathlib import Path


def test_runner_has_no_gt_or_sam2_inference_dependency() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/run_sam2_video_geometry_stability.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "load_camera_mask_ground_truth",
        "load_ground_truth_evaluation_inputs",
        "build_sam2",
        "sam2_video_predictor",
        "nominal_depth_cm",
        "water_level_gt",
        "depth_map_gt",
    ):
        assert forbidden not in source
    assert '"ground_truth_used": False' in source
    assert '"sam2_rerun_count": 0' in source


def test_runner_reuses_existing_configs_without_threshold_override() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/run_sam2_video_geometry_stability.py"
    ).read_text(encoding="utf-8")
    assert "water_surface_aware_mapping.yaml" in source
    assert "water_surface_aware_quality_gate.yaml" in source
    assert '"gate_thresholds_modified": False' in source
    assert "max_boundary_reprojection_p95_px" not in source


def test_runner_verifies_frozen_inputs_before_output_creation() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/run_sam2_video_geometry_stability.py"
    ).read_text(encoding="utf-8")
    verification = source.index("verified = verify_frozen_prediction_inputs(")
    output_creation = source.index("output_root.mkdir", verification)
    assert verification < output_creation
