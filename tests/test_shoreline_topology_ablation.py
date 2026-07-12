import inspect

import numpy as np

from src.evaluation.evaluate_shoreline_topology_ablation import (
    _outer_interface_points,
    apply_topology_method,
    boundary_p50_distance_transform,
    conditional_hole_fill,
    largest_component,
    run_fixed_candidates_before_gt,
    unobservable_component_safety,
)


PARAMETERS = {
    "connectivity": 8,
    "small_component_min_area_px": 100,
    "conditional_hole_max_area_px": 20,
    "conditional_hole_max_water_area_fraction": 0.5,
    "morphological_closing_kernel_px": 3,
    "morphological_closing_iterations": 1,
}


def test_empty_dry_mask_remains_empty_for_every_fixed_method():
    empty = np.zeros((32, 32), dtype=bool)
    methods = [
        "baseline", "largest_component", "small_component_filter", "conditional_hole_fill",
        "morphological_closing", "largest_component_conditional_hole_fill",
        "outer_boundary_only",
        "small_component_filter_conditional_hole_fill", "largest_component_outer_boundary_only",
    ]
    for method in methods:
        repaired, _ = apply_topology_method(empty, empty, method, PARAMETERS)
        assert not np.any(repaired)


def test_largest_component_is_deterministic_and_input_is_not_modified():
    mask = np.zeros((20, 20), dtype=bool); mask[2:8, 2:8] = True; mask[12:15, 12:15] = True
    original = mask.copy(); first = largest_component(mask); second = largest_component(mask)
    assert np.array_equal(mask, original)
    assert np.array_equal(first, second)
    assert np.count_nonzero(first) == 36


def test_conditional_hole_fill_keeps_large_hole_and_fills_small_hole():
    mask = np.ones((20, 20), dtype=bool); mask[3:5, 3:5] = False; mask[8:14, 8:14] = False
    original = mask.copy(); repaired = conditional_hole_fill(mask, 20, 0.5)
    assert np.array_equal(mask, original)
    assert np.all(repaired[3:5, 3:5])
    assert not np.any(repaired[8:14, 8:14])


def test_outer_boundary_excludes_internal_hole_boundary():
    water = np.zeros((30, 30), dtype=bool); water[4:26, 4:26] = True; water[11:19, 11:19] = False
    trusted, points = _outer_interface_points(water, np.zeros_like(water))
    assert points
    assert not np.any(trusted[10:20, 10:20])
    assert np.any(trusted[4, 4:26])


def test_unknown_region_cannot_be_created_as_repaired_camera_evidence():
    mask = np.zeros((20, 20), dtype=bool); mask[4:16, 4:16] = True; mask[8:12, 8:12] = False
    unknown = np.zeros_like(mask); unknown[8:12, 8:12] = True
    repaired, _ = apply_topology_method(mask, unknown, "conditional_hole_fill", PARAMETERS)
    assert not np.any(repaired & unknown)


def test_all_fixed_methods_are_deterministic_and_do_not_modify_input():
    mask = np.zeros((24, 24), dtype=bool); mask[3:20, 3:20] = True; mask[8:10, 8:10] = False
    unknown = np.zeros_like(mask); original = mask.copy()
    methods = [
        "baseline", "largest_component", "small_component_filter", "conditional_hole_fill",
        "morphological_closing", "outer_boundary_only", "largest_component_conditional_hole_fill",
        "small_component_filter_conditional_hole_fill", "largest_component_outer_boundary_only",
    ]
    for method in methods:
        first, _ = apply_topology_method(mask, unknown, method, PARAMETERS)
        second, _ = apply_topology_method(mask, unknown, method, PARAMETERS)
        assert np.array_equal(first, second)
        assert np.array_equal(mask, original)


def test_unobservable_secondary_basin_is_not_added_to_camera_or_dem_prediction():
    component = np.zeros((10, 10), dtype=bool); component[6:9, 6:9] = True
    projected = np.zeros((20, 20), dtype=bool)
    repaired_camera = np.zeros((20, 20), dtype=bool); repaired_camera[2:5, 2:5] = True
    predicted_dem = np.zeros((10, 10), dtype=bool); predicted_dem[1:4, 1:4] = True
    safety = unobservable_component_safety(component, projected, repaired_camera, predicted_dem)
    assert safety["unobservable_safe"] is True
    assert safety["repaired_camera_evidence_overlap_pixels"] == 0
    assert safety["predicted_dem_intersection_cells"] == 0


def test_distance_transform_p50_is_deterministic_for_shifted_boundaries():
    observed = np.zeros((32, 32), dtype=bool); observed[8:24, 8:20] = True
    reprojected = np.zeros_like(observed); reprojected[8:24, 10:22] = True
    first = boundary_p50_distance_transform(observed, np.zeros_like(observed), reprojected)
    second = boundary_p50_distance_transform(observed, np.zeros_like(observed), reprojected)
    assert first is not None
    assert first == second


def test_prediction_candidate_function_has_no_gt_loader_or_evaluation_dependency():
    source = inspect.getsource(run_fixed_candidates_before_gt)
    assert "ground_truth" not in source.lower()
    assert "load_ground_truth" not in source
