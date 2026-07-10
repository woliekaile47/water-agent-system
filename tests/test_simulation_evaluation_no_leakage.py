from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

from src.fusion.project_camera_mask_to_dem import load_prediction_inputs, project_camera_mask_to_dem


def test_prediction_loader_uses_only_allowlisted_inputs(tmp_path: Path):
    dry = tmp_path / "data/simulation/dry/ground_truth"
    wet = tmp_path / "data/simulation/wet/ground_truth"
    config_dir = tmp_path / "simulation/config"
    dry.mkdir(parents=True)
    wet.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    ground_path = dry / "ground_dem_gt.npy"
    np.save(ground_path, np.zeros((2, 2), dtype=np.float32))
    Image.fromarray(np.full((20, 20), 255, dtype=np.uint8)).save(wet / "camera_water_mask_gt.png")
    # These forbidden answer files are deliberately corrupt. Prediction must
    # still run because it must never open them.
    (wet / "dem_water_mask_gt.npy").write_bytes(b"not-a-npy")
    (wet / "depth_map_gt_m.npy").write_bytes(b"not-a-npy")
    (tmp_path / "data/simulation/wet/manifest.json").write_text("not-json", encoding="utf-8")
    sensors = {
        "road": {"length_m": 2.0, "width_m": 2.0, "dem_resolution_m": 1.0},
        "sensor_rig": {"pose_map": {"x_m": -1.0, "y_m": 0.0, "z_m": 0.0}},
        "camera": {"pose_on_rig": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0, "roll_deg": 0.0, "pitch_down_deg": 0.0, "yaw_deg": 0.0}, "width_px": 20, "height_px": 20, "horizontal_fov_deg": 90.0, "near_clip_m": 0.1, "far_clip_m": 10.0},
        "coordinate_frames": {"map": "map", "camera_optical": "camera_optical_frame"},
    }
    (config_dir / "sensors.yaml").write_text(yaml.safe_dump(sensors), encoding="utf-8")
    real_load = np.load
    loaded_paths = []

    def recording_load(path, *args, **kwargs):
        loaded_paths.append(Path(path).name)
        return real_load(path, *args, **kwargs)

    with patch("src.fusion.project_camera_mask_to_dem.np.load", side_effect=recording_load):
        inputs = load_prediction_inputs(tmp_path, "wet", {"dry_case_id": "dry", "sensors_config": "simulation/config/sensors.yaml"})
        predicted, _ = project_camera_mask_to_dem(inputs["ground_dem"], inputs["camera_mask"], inputs["sensors"])
    assert loaded_paths == ["ground_dem_gt.npy"]
    assert predicted.shape == (2, 2)
    assert "water_level_m" not in inputs
    assert "dem_mask" not in inputs and "depth_map" not in inputs
