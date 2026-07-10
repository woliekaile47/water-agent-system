from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from waterlogging_simulation.config import REQUIRED_CASES
from waterlogging_simulation.generator import generate_case
from waterlogging_simulation.geometry import validate_road_obj


SIMULATION_ROOT = Path(__file__).resolve().parents[1]


def _generate(tmp_path: Path, case_id: str):
    manifest = generate_case(
        case_id=case_id,
        project_root=SIMULATION_ROOT.parent,
        config_dir=SIMULATION_ROOT / "config",
        world_template=SIMULATION_ROOT / "worlds" / "low_lying_road.sdf",
        output_root=tmp_path,
    )
    return manifest, tmp_path / case_id


def test_road_obj_is_visual_only_and_structurally_valid(tmp_path):
    manifest, case_dir = _generate(tmp_path, "sim_dry_baseline_001")
    obj_path = case_dir / "metadata" / "road_basin.obj"
    stats = validate_road_obj(obj_path)
    assert stats["vertex_count"] > 0
    assert stats["face_count"] > 0
    assert stats["normal_count"] == stats["vertex_count"]
    assert (case_dir / "metadata" / "road_basin.mtl").exists()

    root = ET.parse(case_dir / "metadata" / "resolved_world.sdf").getroot()
    road_model = root.find(".//model[@name='road_basin']")
    assert road_model is not None
    road_collision = road_model.find("./link/collision[@name='road_collision']")
    road_visual = road_model.find("./link/visual[@name='road_visual']")
    assert road_collision is not None
    assert road_visual is not None
    assert road_collision.find(".//mesh") is None
    assert road_collision.find("./geometry/box") is not None
    assert road_visual.find("./geometry/mesh") is not None

    pose_values = [float(value) for value in road_collision.findtext("pose").split()]
    size_values = [
        float(value) for value in road_collision.findtext("./geometry/box/size").split()
    ]
    collision_top = pose_values[2] + size_values[2] / 2.0
    ground_dem = np.load(case_dir / "ground_truth" / "ground_dem_gt.npy")
    assert collision_top < float(np.min(ground_dem))
    assert manifest["road_collision_mode"] == "simplified_box_below_visual"


def test_water_surface_is_visual_only_without_mesh_collision(tmp_path):
    _, case_dir = _generate(tmp_path, "sim_water_5cm_001")
    root = ET.parse(case_dir / "metadata" / "resolved_world.sdf").getroot()
    water_model = root.find(".//model[@name='water_surface']")
    assert water_model is not None
    assert water_model.find(".//collision") is None
    assert water_model.find(".//mesh") is None
    assert water_model.find("./link/visual/geometry/box") is not None


def test_all_resolved_worlds_pass_fortress_sdf_check(tmp_path):
    ign = shutil.which("ign")
    assert ign is not None, "Gazebo Fortress ign executable is required for this regression test"
    for case_id in sorted(REQUIRED_CASES):
        _, case_dir = _generate(tmp_path, case_id)
        world = case_dir / "metadata" / "resolved_world.sdf"
        result = subprocess.run(
            [ign, "sdf", "-k", str(world)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{case_id}: {result.stdout}\n{result.stderr}"
        assert "Valid." in result.stdout
