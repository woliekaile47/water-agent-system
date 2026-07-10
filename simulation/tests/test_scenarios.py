from pathlib import Path

from waterlogging_simulation.config import (
    REQUIRED_CASES,
    load_configuration,
    resolve_scenario,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_scenarios_yaml_loads_and_has_required_depths():
    sensors, scenarios = load_configuration(CONFIG_DIR)
    assert sensors["simulation"]["ros_distro"] == "humble"
    assert set(scenarios["scenarios"]) == set(REQUIRED_CASES)
    for case_id, depth_cm in REQUIRED_CASES.items():
        assert resolve_scenario(scenarios, case_id)["water_depth_cm"] == depth_cm


def test_two_stage_sensor_enable_policy():
    _, scenarios = load_configuration(CONFIG_DIR)
    dry = resolve_scenario(scenarios, "sim_dry_baseline_001")
    assert dry["lidar_enabled"] is True
    assert dry["camera_enabled"] is True
    for case_id in REQUIRED_CASES:
        scenario = resolve_scenario(scenarios, case_id)
        if scenario["water_depth_cm"] > 0:
            assert scenario["lidar_enabled"] is False
            assert scenario["camera_enabled"] is True


def test_topics_and_frames_are_unique_and_complete():
    sensors, _ = load_configuration(CONFIG_DIR)
    topics = sensors["topic_names"]
    frames = sensors["coordinate_frames"]
    assert len(topics) == len(set(topics.values()))
    assert len(frames) == len(set(frames.values()))
    assert topics["lidar_points"] == "/sim/lidar/points"
    assert sensors["simulation"]["lidar_simulation_mode"] == "deterministic_geometry_generator"
    assert topics["camera_image"] == "/sim/camera/image_raw"
    assert topics["camera_info"] == "/sim/camera/camera_info"
    assert all(
        topic.startswith("/sim/")
        for key, topic in topics.items()
        if key not in {"clock", "tf", "tf_static"}
    )
