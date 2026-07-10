import numpy as np

from src.evaluation.water_surface_aware_quality_gate import evaluate_water_surface_aware_quality_gate


CONFIG = {
    "min_shoreline_intersection_rate": 0.65, "max_camera_mask_edge_touch_ratio": 0.2,
    "min_camera_reprojection_iou": 0.9, "max_boundary_reprojection_p95_px": 3.0,
    "max_candidate_basin_count": 5, "min_water_surface_projection_coverage": 0.95,
    "min_valid_shoreline_samples": 20, "max_shoreline_mad_m": 0.02,
    "max_shoreline_iqr_m": 0.06, "max_physical_depth_m": 0.6,
}


def arguments(intersection_rate=0.9, reprojection_iou=0.98):
    ray = {"shoreline_intersection_success_rate": intersection_rate, "camera_mask_edge_touch_ratio": 0.0}
    shoreline = {"estimated_water_level_m": 0.1, "valid_shoreline_sample_count": 50, "shoreline_height_mad_m": 0.005, "shoreline_height_iqr_m": 0.01, "water_level_converged": True}
    reconstruction = {"candidate_basin_count": 1, "selected_basin_count": 1, "ambiguous_candidate_basins": False, "unobserved_candidate_basin_count": 0, "seed_valid": True}
    consistency = {"camera_reprojection_iou": reprojection_iou, "boundary_reprojection_p95_px": 1.0, "water_surface_projection_coverage": 1.0}
    result = {"max_depth_m": 0.2, "negative_depth_count": 0, "inf_depth_count": 0}
    return ray, shoreline, reconstruction, consistency, result, np.zeros((3, 3), dtype=np.float32)


def test_good_prediction_metrics_pass_but_stay_downstream_ineligible():
    gate = evaluate_water_surface_aware_quality_gate(*arguments(), CONFIG)
    assert gate["status"] == "pass"
    assert gate["global_estimate_status"] == "complete"
    assert gate["observable_region_result_valid"] is True
    assert gate["result_semantics"] == "global_estimate"
    assert gate["area_volume_semantics"] == "complete_estimate"
    assert gate["eligible_for_downstream"] is False


def test_low_shoreline_intersection_rate_rejects():
    gate = evaluate_water_surface_aware_quality_gate(*arguments(intersection_rate=0.2), CONFIG)
    assert gate["status"] == "reject"
    assert "shoreline_intersection_rate_below_threshold" in gate["reasons"]


def test_low_reprojection_iou_rejects_without_ground_truth_input():
    gate = evaluate_water_surface_aware_quality_gate(*arguments(reprojection_iou=0.5), CONFIG)
    assert gate["status"] == "reject"
    assert "camera_reprojection_iou_below_threshold" in gate["reasons"]


def test_unobservable_candidate_is_partial_lower_bound_but_observable_result_valid():
    ray, shoreline, reconstruction, consistency, result, depth = arguments()
    reconstruction.update({
        "candidate_basin_count": 2,
        "selected_basin_count": 1,
        "ambiguous_candidate_basins": True,
        "ambiguous_candidate_basin_count": 1,
        "unobserved_candidate_basin_count": 1,
        "camera_observable_candidate_basin_count": 1,
    })
    gate = evaluate_water_surface_aware_quality_gate(
        ray, shoreline, reconstruction, consistency, result, depth, CONFIG
    )
    assert gate["status"] == "reject"
    assert "ambiguous_candidate_basin" in gate["reasons"]
    assert "candidate_basin_outside_camera_coverage" in gate["reasons"]
    assert gate["global_estimate_status"] == "partial"
    assert gate["observable_region_result_valid"] is True
    assert gate["unobservable_candidate_basin_count"] == 1
    assert gate["ambiguous_candidate_basin_count"] == 1
    assert gate["camera_observable_candidate_basin_count"] == 1
    assert gate["result_semantics"] == "observable_region_estimate"
    assert gate["area_volume_semantics"] == "observable_lower_bound"
    assert gate["eligible_for_downstream"] is False


def test_prediction_gate_signature_has_no_ground_truth_inputs():
    import inspect

    parameters = set(inspect.signature(evaluate_water_surface_aware_quality_gate).parameters)
    assert not parameters.intersection({
        "ground_truth", "true_water_level", "dem_water_mask_gt", "depth_map_gt",
        "true_area", "true_volume",
    })
