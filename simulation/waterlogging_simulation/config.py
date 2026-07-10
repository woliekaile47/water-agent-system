"""Configuration loading and validation for Phase 1 simulation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_CASES = {
    "sim_dry_baseline_001": 0,
    "sim_water_5cm_001": 5,
    "sim_water_10cm_001": 10,
    "sim_water_20cm_001": 20,
    "sim_water_40cm_001": 40,
}

REQUIRED_TOPICS = {
    "lidar_points",
    "camera_image",
    "camera_info",
    "water_level_gt",
    "water_mask_gt",
    "depth_map_gt",
    "clock",
    "tf",
    "tf_static",
}

REQUIRED_FRAMES = {"map", "road", "sensor_mount", "lidar", "camera", "camera_optical"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Simulation config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Simulation config must be a mapping: {config_path}")
    return data


def validate_sensor_config(config: dict[str, Any]) -> None:
    for key in ("simulation", "road", "basin", "sensor_rig", "lidar", "camera", "coordinate_frames", "topic_names"):
        if key not in config:
            raise ValueError(f"sensors.yaml is missing required section: {key}")

    road = config["road"]
    if not 15.0 <= float(road["length_m"]) <= 20.0:
        raise ValueError("road.length_m must be between 15 and 20 m")
    if not 6.0 <= float(road["width_m"]) <= 8.0:
        raise ValueError("road.width_m must be between 6 and 8 m")
    if float(road["dem_resolution_m"]) <= 0 or float(road["mesh_resolution_m"]) <= 0:
        raise ValueError("DEM and mesh resolutions must be positive")
    if config["simulation"].get("lidar_simulation_mode") != "deterministic_geometry_generator":
        raise ValueError("Phase 1A requires lidar_simulation_mode=deterministic_geometry_generator")

    basin = config["basin"]
    if float(basin["half_length_m"]) <= 0 or float(basin["half_width_m"]) <= 0:
        raise ValueError("Basin dimensions must be positive")
    if float(basin["depth_m"]) < 0.40:
        raise ValueError("Basin depth must support the configured 40 cm scene")

    topics = config["topic_names"]
    frames = config["coordinate_frames"]
    missing_topics = REQUIRED_TOPICS.difference(topics)
    missing_frames = REQUIRED_FRAMES.difference(frames)
    if missing_topics:
        raise ValueError(f"Missing topic names: {sorted(missing_topics)}")
    if missing_frames:
        raise ValueError(f"Missing coordinate frames: {sorted(missing_frames)}")
    if len(set(topics.values())) != len(topics):
        raise ValueError("Topic names must be unique")
    if len(set(frames.values())) != len(frames):
        raise ValueError("Coordinate frame names must be unique")
    for name, topic in topics.items():
        if not str(topic).startswith("/"):
            raise ValueError(f"Topic {name} must be absolute: {topic}")
        if name not in {"clock", "tf", "tf_static"} and not str(topic).startswith("/sim/"):
            raise ValueError(f"Simulation topic {name} must use /sim prefix: {topic}")


def validate_scenarios(config: dict[str, Any]) -> None:
    if "defaults" not in config or "scenarios" not in config:
        raise ValueError("scenarios.yaml must contain defaults and scenarios")
    scenarios = config["scenarios"]
    if set(scenarios) != set(REQUIRED_CASES):
        raise ValueError(
            f"Scenario IDs must be exactly {sorted(REQUIRED_CASES)}; got {sorted(scenarios)}"
        )
    for case_id, expected_depth in REQUIRED_CASES.items():
        scenario = scenarios[case_id]
        if int(scenario["water_depth_cm"]) != expected_depth:
            raise ValueError(f"{case_id} water_depth_cm must be {expected_depth}")
        if not isinstance(scenario.get("lidar_enabled"), bool):
            raise ValueError(f"{case_id} lidar_enabled must be boolean")
        if not isinstance(scenario.get("camera_enabled"), bool):
            raise ValueError(f"{case_id} camera_enabled must be boolean")
        if not scenario["camera_enabled"]:
            raise ValueError(f"{case_id} must keep Camera enabled")
        if expected_depth == 0 and not scenario["lidar_enabled"]:
            raise ValueError("Dry baseline must enable LiDAR")
        if expected_depth > 0 and scenario["lidar_enabled"]:
            raise ValueError(f"Water scene {case_id} must disable LiDAR")


def load_configuration(config_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(config_dir).expanduser().resolve()
    sensors = load_yaml(root / "sensors.yaml")
    scenarios = load_yaml(root / "scenarios.yaml")
    validate_sensor_config(sensors)
    validate_scenarios(scenarios)
    return sensors, scenarios


def resolve_scenario(scenarios_config: dict[str, Any], case_id: str) -> dict[str, Any]:
    if case_id not in scenarios_config["scenarios"]:
        raise KeyError(f"Unknown scenario {case_id!r}")
    result = copy.deepcopy(scenarios_config["defaults"])
    result.update(copy.deepcopy(scenarios_config["scenarios"][case_id]))
    result["case_id"] = case_id
    result["random_seed"] = int(result["random_seed"])
    result["water_depth_cm"] = int(result["water_depth_cm"])
    return result
