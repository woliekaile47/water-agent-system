"""Generate resolved Gazebo worlds and deterministic Ground Truth artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .config import load_configuration, resolve_scenario
from .geometry import (
    camera_intrinsics,
    camera_water_mask,
    camera_world_pose,
    dem_grid,
    water_ground_truth,
    write_road_obj,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info(project_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except Exception:
        return "unknown", True


def _gazebo_version() -> str:
    try:
        return subprocess.run(
            ["ign", "gazebo", "--versions"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _material(r: float, g: float, b: float, a: float = 1.0) -> str:
    return f"""
          <material>
            <ambient>{r} {g} {b} {a}</ambient>
            <diffuse>{r} {g} {b} {a}</diffuse>
            <specular>0.15 0.15 0.15 {a}</specular>
          </material>"""


def _road_and_curbs_sdf(road_mesh: Path, sensors: dict[str, Any]) -> str:
    road = sensors["road"]
    length = float(road["length_m"])
    width = float(road["width_m"])
    curb_width = float(road["curb_width_m"])
    curb_height = float(road["curb_height_m"])
    ground_z = -float(sensors["basin"]["depth_m"]) - 0.25
    curb_y = width / 2.0 + curb_width / 2.0
    collision_thickness = 0.04
    conservative_min_road_z = (
        -float(sensors["basin"]["depth_m"])
        - abs(float(road["longitudinal_slope"])) * length / 2.0
        - abs(float(road["cross_slope"])) * width / 2.0
    )
    collision_top_z = conservative_min_road_z - 0.02
    collision_center_z = collision_top_z - collision_thickness / 2.0
    return f"""
    <model name="surrounding_ground">
      <static>true</static>
      <pose>0 0 {ground_z:.6f} 0 0 0</pose>
      <link name="ground_link">
        <collision name="ground_collision"><geometry><box><size>{length + 8.0:.3f} {width + 8.0:.3f} 0.10</size></box></geometry></collision>
        <visual name="ground_visual"><geometry><box><size>{length + 8.0:.3f} {width + 8.0:.3f} 0.10</size></box></geometry>{_material(0.25, 0.32, 0.20)}</visual>
      </link>
    </model>
    <model name="road_basin">
      <static>true</static>
      <link name="road_link">
        <!-- Phase 1A uses a primitive collision below the lowest visual road point. -->
        <collision name="road_collision">
          <pose>0 0 {collision_center_z:.6f} 0 0 0</pose>
          <geometry><box><size>{length:.3f} {width:.3f} {collision_thickness:.3f}</size></box></geometry>
        </collision>
        <visual name="road_visual"><geometry><mesh><uri>file://{road_mesh}</uri></mesh></geometry>{_material(0.18, 0.20, 0.22)}</visual>
      </link>
    </model>
    <model name="curbs">
      <static>true</static>
      <link name="curb_link">
        <collision name="left_curb_collision"><pose>0 {curb_y:.6f} {curb_height / 2.0:.6f} 0 0 0</pose><geometry><box><size>{length:.3f} {curb_width:.3f} {curb_height:.3f}</size></box></geometry></collision>
        <visual name="left_curb_visual"><pose>0 {curb_y:.6f} {curb_height / 2.0:.6f} 0 0 0</pose><geometry><box><size>{length:.3f} {curb_width:.3f} {curb_height:.3f}</size></box></geometry>{_material(0.60, 0.60, 0.58)}</visual>
        <collision name="right_curb_collision"><pose>0 {-curb_y:.6f} {curb_height / 2.0:.6f} 0 0 0</pose><geometry><box><size>{length:.3f} {curb_width:.3f} {curb_height:.3f}</size></box></geometry></collision>
        <visual name="right_curb_visual"><pose>0 {-curb_y:.6f} {curb_height / 2.0:.6f} 0 0 0</pose><geometry><box><size>{length:.3f} {curb_width:.3f} {curb_height:.3f}</size></box></geometry>{_material(0.60, 0.60, 0.58)}</visual>
      </link>
    </model>"""


def _water_sdf(water_level_m: float | None, sensors: dict[str, Any]) -> str:
    if water_level_m is None:
        return "<!-- dry scene: no water surface model -->"
    road = sensors["road"]
    return f"""
    <model name="water_surface">
      <static>true</static>
      <pose>0 0 {water_level_m:.8f} 0 0 0</pose>
      <link name="water_link">
        <visual name="water_visual">
          <geometry><box><size>{float(road['length_m']):.3f} {float(road['width_m']):.3f} 0.004</size></box></geometry>
          <material>
            <ambient>0.05 0.28 0.48 0.55</ambient>
            <diffuse>0.08 0.40 0.70 0.55</diffuse>
            <specular>0.90 0.90 0.95 0.55</specular>
          </material>
          <transparency>0.45</transparency>
          <cast_shadows>false</cast_shadows>
        </visual>
      </link>
    </model>"""


def _sensor_rig_sdf(scenario: dict[str, Any], sensors: dict[str, Any]) -> str:
    rig = sensors["sensor_rig"]
    rig_pose = rig["pose_map"]
    camera = sensors["camera"]
    camera_pose = camera["pose_on_rig"]
    lidar = sensors["lidar"]
    lidar_pose = lidar["pose_on_rig"]
    topics = sensors["topic_names"]
    frames = sensors["coordinate_frames"]
    camera_block = ""
    if scenario["camera_enabled"]:
        camera_block = f"""
        <sensor name="sim_rgb_camera" type="camera">
          <pose>{camera_pose['x_m']} {camera_pose['y_m']} {camera_pose['z_m']} {math.radians(float(camera_pose['roll_deg'])):.8f} {math.radians(float(camera_pose['pitch_down_deg'])):.8f} {math.radians(float(camera_pose['yaw_deg'])):.8f}</pose>
          <camera>
            <horizontal_fov>{math.radians(float(camera['horizontal_fov_deg'])):.8f}</horizontal_fov>
            <image><width>{int(camera['width_px'])}</width><height>{int(camera['height_px'])}</height></image>
            <clip><near>{float(camera['near_clip_m'])}</near><far>{float(camera['far_clip_m'])}</far></clip>
          </camera>
          <always_on>true</always_on>
          <update_rate>{float(camera['update_rate_hz'])}</update_rate>
          <visualize>true</visualize>
          <topic>{topics['camera_image']}</topic>
          <camera_info_topic>{topics['camera_info']}</camera_info_topic>
          <gz_frame_id>{frames['camera_optical']}</gz_frame_id>
        </sensor>"""
    lidar_block = ""
    if scenario["lidar_enabled"]:
        lidar_block = f"""
        <!-- Phase 1A appearance only; PointCloud2 comes from deterministic geometry. -->
        <visual name="sim_multiline_lidar_visual">
          <pose>{lidar_pose['x_m']} {lidar_pose['y_m']} {lidar_pose['z_m']} {math.radians(float(lidar_pose['roll_deg'])):.8f} {math.radians(float(lidar_pose['pitch_deg'])):.8f} {math.radians(float(lidar_pose['yaw_deg'])):.8f}</pose>
          <geometry><cylinder><radius>0.09</radius><length>0.12</length></cylinder></geometry>
          {_material(0.08, 0.08, 0.10)}
        </visual>"""
    return f"""
    <model name="roadside_sensor_rig">
      <static>true</static>
      <pose>{rig_pose['x_m']} {rig_pose['y_m']} {rig_pose['z_m']} {math.radians(float(rig_pose['roll_deg'])):.8f} {math.radians(float(rig_pose['pitch_deg'])):.8f} {math.radians(float(rig_pose['yaw_deg'])):.8f}</pose>
      <link name="sensor_mount_link">
        <collision name="pole_collision"><pose>0 0 {float(rig['pole_height_m']) / 2.0:.6f} 0 0 0</pose><geometry><cylinder><radius>0.07</radius><length>{float(rig['pole_height_m']):.3f}</length></cylinder></geometry></collision>
        <visual name="pole_visual"><pose>0 0 {float(rig['pole_height_m']) / 2.0:.6f} 0 0 0</pose><geometry><cylinder><radius>0.07</radius><length>{float(rig['pole_height_m']):.3f}</length></cylinder></geometry>{_material(0.35, 0.36, 0.38)}</visual>
        {camera_block}
        {lidar_block}
      </link>
    </model>"""


def render_world(
    template_path: Path,
    output_path: Path,
    road_mesh: Path,
    scenario: dict[str, Any],
    sensors: dict[str, Any],
    water_level_m: float | None,
) -> Path:
    template = template_path.read_text(encoding="utf-8")
    models = "\n".join(
        [
            _road_and_curbs_sdf(road_mesh, sensors),
            _water_sdf(water_level_m, sensors),
            _sensor_rig_sdf(scenario, sensors),
        ]
    )
    rendered = template.replace("@PHYSICS_STEP@", str(sensors["simulation"]["physics_step_s"]))
    rendered = rendered.replace("@GENERATED_MODELS@", models)
    if "@" in rendered:
        raise RuntimeError("Unresolved token remains in generated SDF")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def generate_case(
    case_id: str,
    project_root: str | Path,
    config_dir: str | Path,
    world_template: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve()
    config_path = Path(config_dir).expanduser().resolve()
    sensors, scenarios = load_configuration(config_path)
    scenario = resolve_scenario(scenarios, case_id)
    np.random.seed(int(scenario["random_seed"]))

    simulation_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else project / "data" / "simulation"
    )
    case_dir = simulation_root / case_id
    gt_dir = case_dir / "ground_truth"
    metadata_dir = case_dir / "metadata"
    rosbag_dir = case_dir / "rosbag"
    for directory in (gt_dir, metadata_dir, rosbag_dir):
        directory.mkdir(parents=True, exist_ok=True)

    xx, yy, ground_dem = dem_grid(sensors)
    water_level_m, dem_mask, depth_map, area_m2, volume_m3 = water_ground_truth(
        ground_dem, scenario, sensors
    )
    camera_mask = camera_water_mask(xx, yy, dem_mask, water_level_m, sensors)

    paths = {
        "ground_dem": gt_dir / "ground_dem_gt.npy",
        "dem_mask": gt_dir / "dem_water_mask_gt.npy",
        "depth_map": gt_dir / "depth_map_gt_m.npy",
        "camera_mask": gt_dir / "camera_water_mask_gt.png",
        "ground_truth_metadata": gt_dir / "ground_truth_metadata.json",
        "road_mesh": metadata_dir / "road_basin.obj",
        "road_material": metadata_dir / "road_basin.mtl",
        "resolved_world": metadata_dir / "resolved_world.sdf",
        "config_snapshot": metadata_dir / "config_snapshot.yaml",
        "manifest": case_dir / "manifest.json",
    }
    np.save(paths["ground_dem"], ground_dem.astype(np.float32))
    np.save(paths["dem_mask"], dem_mask.astype(bool))
    np.save(paths["depth_map"], depth_map.astype(np.float32))
    Image.fromarray(camera_mask, mode="L").save(paths["camera_mask"])
    write_road_obj(paths["road_mesh"], sensors)
    render_world(
        Path(world_template).expanduser().resolve(),
        paths["resolved_world"],
        paths["road_mesh"],
        scenario,
        sensors,
        water_level_m,
    )
    config_snapshot = {"scenario": scenario, "sensors": sensors}
    paths["config_snapshot"].write_text(
        yaml.safe_dump(config_snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    gt_metadata = {
        "schema_version": "1.0",
        "data_role": "ground_truth",
        "case_id": case_id,
        "formula": "depth_gt(x,y)=max(0,water_level_gt-ground_dem_gt(x,y))",
        "water_level_m": water_level_m,
        "water_depth_cm": int(scenario["water_depth_cm"]),
        "dem_shape_rc": [int(ground_dem.shape[0]), int(ground_dem.shape[1])],
        "dem_resolution_m": float(sensors["road"]["dem_resolution_m"]),
        "wet_cell_count": int(np.count_nonzero(dem_mask)),
        "water_area_m2": area_m2,
        "water_volume_m3": volume_m3,
        "camera_mask_pixel_count": int(np.count_nonzero(camera_mask)),
        "note": "Simulation Ground Truth only; this is not an algorithm prediction.",
    }
    paths["ground_truth_metadata"].write_text(
        json.dumps(gt_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    git_commit, git_dirty = _git_info(project)
    manifest = {
        "schema_version": "1.0",
        "data_role": "ground_truth",
        "case_id": case_id,
        "scenario_type": scenario["scenario_type"],
        "simulator": sensors["simulation"]["simulator"],
        "ros_distro": os.environ.get("ROS_DISTRO", sensors["simulation"]["ros_distro"]),
        "gazebo_version": _gazebo_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "water_depth_cm": int(scenario["water_depth_cm"]),
        "water_level_m": water_level_m,
        "water_level_mode": scenario["water_level_mode"],
        "lidar_enabled": bool(scenario["lidar_enabled"]),
        "lidar_simulation_mode": sensors["simulation"]["lidar_simulation_mode"],
        "camera_enabled": bool(scenario["camera_enabled"]),
        "sensor_pose": {
            "rig": sensors["sensor_rig"]["pose_map"],
            "lidar_on_rig": sensors["lidar"]["pose_on_rig"],
            "camera_on_rig": sensors["camera"]["pose_on_rig"],
            "camera_world": camera_world_pose(sensors),
        },
        "camera_intrinsics": camera_intrinsics(sensors),
        "coordinate_frames": sensors["coordinate_frames"],
        "topic_names": sensors["topic_names"],
        "ground_dem_path": _relative(paths["ground_dem"], project),
        "camera_mask_gt_path": _relative(paths["camera_mask"], project),
        "dem_mask_gt_path": _relative(paths["dem_mask"], project),
        "depth_map_gt_path": _relative(paths["depth_map"], project),
        "ground_truth_metadata_path": _relative(paths["ground_truth_metadata"], project),
        "resolved_world_path": _relative(paths["resolved_world"], project),
        "road_visual_mesh_path": _relative(paths["road_mesh"], project),
        "road_material_path": _relative(paths["road_material"], project),
        "road_collision_mode": "simplified_box_below_visual",
        "rosbag_output_dir": _relative(rosbag_dir, project),
        "water_area_m2": area_m2,
        "water_volume_m3": volume_m3,
        "random_seed": int(scenario["random_seed"]),
        "config_snapshot": config_snapshot,
        "config_snapshot_path": _relative(paths["config_snapshot"], project),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "artifact_sha256": {
            key: _sha256(paths[key])
            for key in (
                "ground_dem",
                "camera_mask",
                "dem_mask",
                "depth_map",
                "road_mesh",
                "road_material",
                "resolved_world",
            )
        },
        "prediction_artifacts": None,
        "note": "All files under ground_truth are simulation truth and must not be presented as predictions.",
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
