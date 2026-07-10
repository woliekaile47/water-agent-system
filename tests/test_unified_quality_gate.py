import numpy as np

from src.evaluation.unified_water_quality_gate import evaluate_quality_gate


CONFIG = {
    "min_projection_coverage": 0.4, "max_component_count": 4,
    "min_largest_component_ratio": 0.8, "min_boundary_samples": 20,
    "max_boundary_mad_m": 0.025, "max_boundary_iqr_m": 0.075,
    "require_valid_inner_outer_bracket": True, "max_physical_depth_m": 0.6,
}


def inputs(boundary_count=30, mad=0.01, iqr=0.02):
    camera = np.ones((4, 4), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=bool)
    depth = np.full((4, 4), 0.1, dtype=np.float32)
    projection = {"projection_coverage": 0.8}
    boundary = {"component_count": 1, "largest_component_ratio": 1.0, "valid_boundary_sample_count": boundary_count, "boundary_height_mad_m": mad, "boundary_height_iqr_m": iqr, "all_components_bracket_valid": True, "estimated_water_level_m": 0.1}
    result = {"max_depth_m": 0.1, "negative_depth_count": 0, "water_area_m2": 1.0, "water_volume_m3": 0.1, "inf_depth_count": 0}
    return camera, mask, depth, projection, boundary, result


def test_good_diagnostics_pass_but_never_enable_downstream():
    gate = evaluate_quality_gate(*inputs(), CONFIG)
    assert gate["status"] == "pass"
    assert gate["reasons"] == []
    assert gate["eligible_for_downstream"] is False


def test_too_few_boundary_samples_rejects():
    gate = evaluate_quality_gate(*inputs(boundary_count=3), CONFIG)
    assert gate["status"] == "reject"
    assert "boundary_sample_count_below_threshold" in gate["reasons"]


def test_excessive_boundary_dispersion_rejects():
    gate = evaluate_quality_gate(*inputs(mad=0.2, iqr=0.3), CONFIG)
    assert gate["status"] == "reject"
    assert "boundary_height_mad_above_threshold" in gate["reasons"]
    assert "boundary_height_iqr_above_threshold" in gate["reasons"]
