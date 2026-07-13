import inspect
from pathlib import Path

import numpy as np

from scripts.run_sam2_shoreline_geometry_diagnostic import build_parser
from src.fusion.project_camera_mask_to_dem import camera_model
from src.fusion.sam2_shoreline_geometry_adapter import (
    deterministic_seed_pixels,
    extended_reprojection_consistency,
    height_statistics,
    intersect_pixel_rays,
    optical_ray_direction,
    outer_boundary_mask,
    prediction_semantics,
    ray_hits_to_dem_seed_mask,
    reconstruct_seed_connected_lowland,
)


def sensors():
    return {
        "road": {"length_m": 4.0, "width_m": 4.0, "dem_resolution_m": 1.0},
        "sensor_rig": {"pose_map": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0}},
        "camera": {
            "pose_on_rig": {
                "x_m": 0.0, "y_m": 0.0, "z_m": 2.0, "roll_deg": 0.0,
                "pitch_down_deg": 90.0, "yaw_deg": 0.0,
            },
            "width_px": 100, "height_px": 80, "horizontal_fov_deg": 90.0,
            "near_clip_m": 0.1, "far_clip_m": 10.0,
        },
        "coordinate_frames": {"map": "map", "camera_optical": "camera_optical_frame"},
    }


RAY_CONFIG = {
    "ray_min_m": 0.1,
    "ray_max_m": 10.0,
    "ray_step_m": 0.1,
    "bisection_iterations": 20,
}


def test_sampled_camera_ray_uses_optical_z_forward_and_phase2b_intersection():
    model = camera_model(sensors())
    optical = optical_ray_direction(model["cx"], model["cy"], model)
    assert np.allclose(optical, [0.0, 0.0, 1.0])
    records, diagnostics = intersect_pixel_rays(
        np.asarray([[model["cx"], model["cy"]]]),
        np.zeros((4, 4), dtype=np.float32),
        sensors(),
        RAY_CONFIG,
        "test",
    )
    assert diagnostics["successful_intersection_count"] == 1
    assert records[0]["hit_status"] == "success"
    assert records[0]["iteration_count"] == 20
    assert abs(records[0]["intersection_residual_m"]) < 1e-6
    repeated_records, repeated_diagnostics = intersect_pixel_rays(
        np.asarray([[model["cx"], model["cy"]]]),
        np.zeros((4, 4), dtype=np.float32),
        sensors(),
        RAY_CONFIG,
        "test",
    )
    assert repeated_records == records
    assert repeated_diagnostics == diagnostics


def test_seed_pixels_are_inside_mask_repeatable_and_map_to_dem():
    mask = np.zeros((80, 100), dtype=bool)
    mask[30:50, 40:60] = True
    positive = np.asarray([[45.0, 35.0], [55.0, 45.0], [1.0, 1.0]])
    first, first_sources = deterministic_seed_pixels(mask, positive, maximum_mask_seed_count=8)
    second, second_sources = deterministic_seed_pixels(mask, positive, maximum_mask_seed_count=8)
    assert np.array_equal(first, second)
    assert first_sources == second_sources
    assert all(mask[int(round(v)), int(round(u))] for u, v in first)
    model = camera_model(sensors())
    records, _ = intersect_pixel_rays(
        np.asarray([[model["cx"], model["cy"]]]), np.zeros((4, 4)), sensors(), RAY_CONFIG, "seed"
    )
    seed_mask, mapped = ray_hits_to_dem_seed_mask(records, (4, 4), sensors())
    assert np.count_nonzero(seed_mask) == 1
    assert mapped[0]["dem_cell_valid"] is True


def test_seed_connected_reconstruction_does_not_add_independent_basin():
    dem = np.zeros((4, 4), dtype=np.float32)
    dem[:, 2] = 1.0
    seed = np.zeros_like(dem, dtype=bool)
    seed[1, 0] = True
    camera_mask = np.ones((80, 100), dtype=bool)
    predicted, diagnostics = reconstruct_seed_connected_lowland(
        dem,
        0.5,
        seed,
        {"connectivity": 4, "lowland_margin_m": 0.0, "min_seed_cells_per_basin": 1},
        camera_mask,
        sensors(),
    )
    assert np.all(predicted[:, :2])
    assert not np.any(predicted[:, 2:])
    assert diagnostics["candidate_basin_count"] == 2
    assert diagnostics["selected_basin_count"] == 1
    assert diagnostics["ambiguous_candidate_basin_count"] == 1


def test_outer_boundary_excludes_enclosed_hole_boundary():
    mask = np.zeros((9, 9), dtype=bool)
    mask[1:8, 1:8] = True
    mask[4, 4] = False
    outer = outer_boundary_mask(mask)
    assert np.count_nonzero(outer) == 24
    assert not outer[3, 4]
    assert not outer[4, 3]
    assert not outer[4, 5]
    assert not outer[5, 4]


def test_height_statistics_include_requested_percentiles():
    stats = height_statistics(np.arange(1.0, 11.0))
    assert stats["count"] == 10
    assert stats["p10_m"] is not None
    assert stats["p25_m"] is not None
    assert stats["p90_m"] is not None
    assert stats["MAD_m"] == 2.5


def test_bool_manual_mask_is_converted_for_phase2b_reprojection_metrics():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:22] = True
    metrics = extended_reprojection_consistency(
        mask,
        np.where(mask, 255, 0).astype(np.uint8),
        {"water_surface_projection_coverage": 1.0},
    )
    assert metrics["camera_reprojection_iou"] == 1.0
    assert metrics["camera_reprojection_precision"] == 1.0
    assert metrics["camera_reprojection_recall"] == 1.0
    assert metrics["boundary_reprojection_p50_px"] == 0.0
    assert metrics["outer_boundary_reprojection_p95_px"] == 0.0


def test_prediction_cli_and_modules_have_no_water_state_answer_inputs():
    options = {option for action in build_parser()._actions for option in action.option_strings}
    assert not options.intersection({"--gt-mask", "--gt-water-level", "--known-depth"})
    adapter_source = inspect.getsource(inspect.getmodule(prediction_semantics))
    runner_source = Path("scripts/run_sam2_shoreline_geometry_diagnostic.py").read_text(encoding="utf-8")
    combined = adapter_source + runner_source
    for forbidden in (
        "load_ground_truth_evaluation_inputs",
        "camera_water_mask_gt",
        "depth_map_gt",
        "water_level_gt",
        "nominal_depth_cm",
    ):
        assert forbidden not in combined
    assert 'json.loads(paths["source_trace_custody_only"]' not in adapter_source


def test_output_semantics_are_non_authoritative_and_downstream_ineligible():
    first = prediction_semantics()
    second = prediction_semantics()
    assert first == second
    assert first["ground_truth_used"] is False
    assert first["authoritative"] is False
    assert first["eligible_for_formal_s5_s8"] is False
    assert first["eligible_for_downstream"] is False
    assert first["result_scope"] == "single_frame_camera_visible_region"
