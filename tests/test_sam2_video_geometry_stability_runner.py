"""Protocol tests for the C7-3 video geometry runner."""

import json
from pathlib import Path

import numpy as np

from scripts.run_sam2_video_geometry_stability import json_compatible


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


def test_runner_mask_stabilization_is_optional_and_auditable() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/run_sam2_video_geometry_stability.py"
    ).read_text(encoding="utf-8")
    assert '"--mask-stabilization-config"' in source
    assert "disabled_raw_mask_passthrough" in source
    assert "stabilize_frozen_mask_sequence" in source
    assert '"raw_mask_area_pixels"' in source
    assert '"stabilized_mask_area_pixels"' in source
    assert '"raw_stabilized_mask_iou"' in source
    assert '"raw_mask_sha256"' in source
    assert '"stabilized_mask_content_sha256"' in source
    assert '"geometry_input_mask_content_sha256"' in source
    assert '"anchor_stabilized_mask_npy_sha256"' in source
    assert '"anchor_stabilized_mask_content_sha256"' in source
    assert '"mask_stabilization_config_sha256"' in source
    assert '"mask_stabilization_ground_truth_used": False' in source
    assert "phase2d_c7_video_geometry_stability_v1_mask_stabilization_v1" in source


def test_runner_stabilization_does_not_change_gate_or_run_sam2() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/run_sam2_video_geometry_stability.py"
    ).read_text(encoding="utf-8")
    assert '"gate_thresholds_modified": False' in source
    assert "build_sam2" not in source
    assert "sam2_video_predictor" not in source
    assert "max_boundary_reprojection_p95_px" not in source
    assert "camera_water_mask_gt" not in source
    assert "water_level_gt" not in source


def test_runner_preserves_legacy_mask_field_only_for_disabled_passthrough() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/run_sam2_video_geometry_stability.py"
    ).read_text(encoding="utf-8")
    disabled = source.index("if stabilization_config is None:", source.index("row.update({"))
    legacy_field = source.index('row["mask_sha256"] = frame["mask_sha256"]', disabled)
    enabled = source.index("else:", legacy_field)
    raw_audit_field = source.index('"raw_mask_sha256": frame["mask_sha256"]', enabled)
    assert disabled < legacy_field < enabled < raw_audit_field


def test_runner_omits_nested_ndarray_payloads_from_diagnostic_json() -> None:
    value = {
        "scalar": np.float32(1.25),
        "nested": {
            "diagnostic_mask": np.zeros((3, 4), dtype=np.uint8),
        },
    }
    safe = json_compatible(value)
    assert safe["scalar"] == 1.25
    assert safe["nested"]["diagnostic_mask"] == {
        "omitted_value_type": "numpy.ndarray",
        "shape": [3, 4],
        "dtype": "uint8",
    }
    json.dumps(safe, allow_nan=False)
