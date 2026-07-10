from pathlib import Path

import json
import numpy as np
import pytest

from waterlogging_simulation.config import REQUIRED_CASES
from waterlogging_simulation.generator import generate_case


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MANIFEST_FIELDS = {
    "case_id",
    "scenario_type",
    "simulator",
    "ros_distro",
    "gazebo_version",
    "created_at",
    "water_depth_cm",
    "water_level_m",
    "lidar_enabled",
    "lidar_simulation_mode",
    "camera_enabled",
    "sensor_pose",
    "camera_intrinsics",
    "coordinate_frames",
    "topic_names",
    "ground_dem_path",
    "camera_mask_gt_path",
    "dem_mask_gt_path",
    "depth_map_gt_path",
    "road_visual_mesh_path",
    "road_collision_mode",
    "water_area_m2",
    "water_volume_m3",
    "random_seed",
    "config_snapshot",
    "git_commit",
}


@pytest.mark.parametrize("case_id", sorted(REQUIRED_CASES))
def test_manifest_and_artifacts(tmp_path, case_id):
    manifest = generate_case(
        case_id=case_id,
        project_root=SIMULATION_ROOT.parent,
        config_dir=SIMULATION_ROOT / "config",
        world_template=SIMULATION_ROOT / "worlds" / "low_lying_road.sdf",
        output_root=tmp_path,
    )
    assert REQUIRED_MANIFEST_FIELDS.issubset(manifest)
    assert manifest["data_role"] == "ground_truth"
    assert manifest["prediction_artifacts"] is None
    assert manifest["lidar_simulation_mode"] == "deterministic_geometry_generator"
    assert manifest["road_collision_mode"] == "simplified_box_below_visual"
    case_dir = tmp_path / case_id
    stored = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stored["case_id"] == case_id
    assert "@" not in (case_dir / "metadata" / "resolved_world.sdf").read_text(encoding="utf-8")
    depth = np.load(case_dir / "ground_truth" / "depth_map_gt_m.npy")
    mask = np.load(case_dir / "ground_truth" / "dem_water_mask_gt.npy")
    assert np.all(depth >= 0.0)
    assert np.all(depth[~mask] == 0.0)
