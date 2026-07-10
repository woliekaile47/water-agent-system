import numpy as np

from src.fusion.project_camera_mask_to_dem import camera_model
from src.fusion.water_surface_aware_mask_to_dem import camera_ray_map, intersect_ray_with_dem


def sensors():
    return {
        "road": {"length_m": 4.0, "width_m": 4.0, "dem_resolution_m": 1.0},
        "sensor_rig": {"pose_map": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0}},
        "camera": {
            "pose_on_rig": {"x_m": 0.0, "y_m": 0.0, "z_m": 2.0, "roll_deg": 0.0, "pitch_down_deg": 90.0, "yaw_deg": 0.0},
            "width_px": 100, "height_px": 80, "horizontal_fov_deg": 90.0,
            "near_clip_m": 0.1, "far_clip_m": 10.0,
        },
        "coordinate_frames": {"map": "map", "camera_optical": "camera_optical_frame"},
    }


CONFIG = {"ray_min_m": 0.1, "ray_max_m": 10.0, "ray_step_m": 0.1, "bisection_iterations": 20}


def test_camera_center_ray_uses_optical_z_forward():
    model = camera_model(sensors())
    origin, direction = camera_ray_map(model["cx"], model["cy"], model)
    assert np.allclose(origin, [0.0, 0.0, 2.0])
    assert np.allclose(direction, [0.0, 0.0, -1.0], atol=1e-12)


def test_ray_intersects_flat_dem():
    hit, reason = intersect_ray_with_dem(np.asarray([0.0, 0.0, 2.0]), np.asarray([0.0, 0.0, -1.0]), np.zeros((4, 4)), sensors(), CONFIG)
    assert reason == "success"
    assert hit is not None
    assert abs(hit["z_ray_m"]) < 1e-6
    assert abs(hit["residual_m"]) < 1e-6


def test_ray_intersects_sloped_dem_repeatably():
    xs = np.asarray([-1.5, -0.5, 0.5, 1.5])
    dem = np.tile(0.1 * xs, (4, 1))
    args = (np.asarray([0.0, 0.0, 2.0]), np.asarray([0.2, 0.0, -1.0]), dem, sensors(), CONFIG)
    first = intersect_ray_with_dem(*args)
    second = intersect_ray_with_dem(*args)
    assert first == second
    assert first[0] is not None and first[1] == "success"
    assert abs(first[0]["residual_m"]) < 1e-6


def test_ray_outside_map_or_without_crossing_is_safe():
    hit, reason = intersect_ray_with_dem(np.asarray([10.0, 10.0, 2.0]), np.asarray([0.0, 0.0, -1.0]), np.zeros((4, 4)), sensors(), CONFIG)
    assert hit is None
    assert reason == "ray_missed_dem_bounds"
