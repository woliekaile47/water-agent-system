import numpy as np

from src.fusion.project_camera_mask_to_dem import (
    map_points_to_camera_optical,
    camera_model,
    project_camera_mask_to_dem,
)


def sensors():
    return {
        "road": {"length_m": 2.0, "width_m": 2.0, "dem_resolution_m": 1.0},
        "sensor_rig": {"pose_map": {"x_m": -1.0, "y_m": 0.0, "z_m": 0.0}},
        "camera": {
            "pose_on_rig": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0, "roll_deg": 0.0, "pitch_down_deg": 0.0, "yaw_deg": 0.0},
            "width_px": 100, "height_px": 80, "horizontal_fov_deg": 90.0,
            "near_clip_m": 0.1, "far_clip_m": 10.0,
        },
        "coordinate_frames": {"map": "map", "camera_optical": "camera_optical_frame"},
    }


def test_optical_frame_axes_and_projection_are_correct():
    model = camera_model(sensors())
    optical = map_points_to_camera_optical(np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]), model)
    assert np.allclose(optical[0], [0.0, 0.0, 2.0])
    assert optical[1, 0] > 0.0  # map -Y is optical right
    assert model["frame_id"] == "camera_optical_frame"


def test_out_of_image_points_are_filtered_and_repeatable():
    dem = np.zeros((2, 2), dtype=np.float32)
    mask = np.full((80, 100), 255, dtype=np.uint8)
    first, diagnostics = project_camera_mask_to_dem(dem, mask, sensors())
    second, diagnostics_again = project_camera_mask_to_dem(dem, mask, sensors())
    assert np.array_equal(first, second)
    assert diagnostics == diagnostics_again
    assert diagnostics["invalid_reasons"]["outside_image"] > 0
    assert np.count_nonzero(first) == diagnostics["projected_dem_cell_count"]
