"""Tests for the transport-neutral, fail-closed sensor input contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from src.sensors.unified_sensor_input import (
    CAMERA_IMAGE_TYPE,
    POINT_CLOUD_TYPE,
    SensorInputContract,
    build_readiness_report,
    evaluate_observation,
    load_sensor_input_contract,
    required_topic_contract,
    validate_calibration,
)


def make_contract(**overrides) -> SensorInputContract:
    values = {
        "schema_version": 1,
        "input_mode": "rosbag",
        "operating_phase": "combined_validation",
        "camera_topic": "/hik_camera/image_raw",
        "lidar_topic": "/cx/lslidar_point_cloud",
        "camera_info_topic": "/hik_camera/camera_info",
        "tf_static_topic": "/tf_static",
        "expected_camera_frame": "hik_camera_link",
        "expected_lidar_frame": "laser_link",
        "expected_image_width": 640,
        "expected_image_height": 360,
        "bag_path": "/tmp/example_bag",
        "live_observation_timeout_seconds": 5.0,
        "calibration_path": None,
        "calibration_required_for_geometry": True,
    }
    values.update(overrides)
    return SensorInputContract(**values)


def passing_observation(contract: SensorInputContract) -> dict:
    return {
        "transport": contract.input_mode,
        "topic_types": {
            contract.camera_topic: CAMERA_IMAGE_TYPE,
            contract.lidar_topic: POINT_CLOUD_TYPE,
        },
        "message_counts": {
            contract.camera_topic: 12,
            contract.lidar_topic: 12,
        },
        "samples": {
            contract.camera_topic: {
                "frame_id": "hik_camera_link",
                "width": 640,
                "height": 360,
                "encoding": "bgr8",
            },
            contract.lidar_topic: {
                "frame_id": "laser_link",
                "fields": ["x", "y", "z", "intensity"],
            },
        },
    }


def write_config(path: Path, *, ground_truth_key: bool = False) -> None:
    data = {
        "schema_version": 1,
        "sensor_input": {
            "input_mode": "rosbag",
            "operating_phase": "combined_validation",
            "topics": {
                "camera_image": "/hik_camera/image_raw",
                "lidar_points": "/cx/lslidar_point_cloud",
                "camera_info": "/hik_camera/camera_info",
                "tf_static": "/tf_static",
            },
            "expected_frames": {
                "camera": "hik_camera_link",
                "lidar": "laser_link",
            },
            "expected_image": {"width": 640, "height": 360},
            "sources": {
                "rosbag": {"path": "/tmp/example_bag"},
                "live_topics": {"observation_timeout_seconds": 5.0},
            },
        },
        "calibration": {"path": None, "required_for_geometry": True},
        "safety": {
            "allow_ground_truth": False,
            "allow_depth_inversion_without_calibration": False,
        },
    }
    if ground_truth_key:
        data["sensor_input"]["ground_truth_path"] = "/forbidden"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_valid_calibration(path: Path) -> None:
    data = {
        "schema_version": 1,
        "frames": {
            "map": "map",
            "camera": "hik_camera_link",
            "camera_optical": "hik_camera_optical_frame",
            "lidar": "laser_link",
        },
        "camera": {
            "image_width": 640,
            "image_height": 360,
            "k": [500.0, 0.0, 320.0, 0.0, 500.0, 180.0, 0.0, 0.0, 1.0],
            "d": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "transforms": {
            "map_from_camera_optical": {
                "translation_xyz": [0.0, 0.0, 2.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "map_from_lidar": {
                "translation_xyz": [0.0, 0.0, 2.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_mode_override_preserves_the_same_downstream_contract(tmp_path: Path) -> None:
    config = tmp_path / "input.yaml"
    write_config(config)
    offline = load_sensor_input_contract(config)
    live = load_sensor_input_contract(config, input_mode="live_topics")
    assert offline.input_mode == "rosbag"
    assert live.input_mode == "live_topics"
    for field in (
        "camera_topic",
        "lidar_topic",
        "camera_info_topic",
        "tf_static_topic",
        "expected_camera_frame",
        "expected_lidar_frame",
        "expected_image_width",
        "expected_image_height",
    ):
        assert getattr(offline, field) == getattr(live, field)


def test_missing_calibration_blocks_geometry_not_valid_ingest() -> None:
    contract = make_contract()
    observation = passing_observation(contract)
    calibration = validate_calibration(contract)
    report = build_readiness_report(contract, observation, calibration)
    assert report["data_ingest_status"] == "pass"
    assert report["calibration"]["status"] == "missing"
    assert report["geometry_readiness"] == "blocked"
    assert report["depth_inversion_allowed"] is False
    assert report["next_allowed_stage"] == "data_ingest_only"
    assert "independent_calibration_not_ready" in report["blocked_reasons"]
    assert report["ground_truth_used"] is False
    assert report["eligible_for_downstream"] is False


def test_valid_independent_calibration_allows_geometry_readiness(tmp_path: Path) -> None:
    calibration_path = tmp_path / "real_calibration.yaml"
    write_valid_calibration(calibration_path)
    contract = make_contract(calibration_path=str(calibration_path))
    report = build_readiness_report(
        contract,
        passing_observation(contract),
        validate_calibration(contract),
    )
    assert report["calibration"]["status"] == "valid"
    assert report["geometry_readiness"] == "ready"
    assert report["depth_inversion_allowed"] is True
    assert report["eligible_for_downstream"] is False


def test_ground_truth_fields_are_rejected_from_prediction_config(tmp_path: Path) -> None:
    config = tmp_path / "forbidden.yaml"
    write_config(config, ground_truth_key=True)
    with pytest.raises(ValueError, match="Ground Truth"):
        load_sensor_input_contract(config)


def test_topic_type_mismatch_rejects_ingest() -> None:
    contract = make_contract()
    observation = passing_observation(contract)
    observation["topic_types"][contract.camera_topic] = "std_msgs/msg/String"
    result = evaluate_observation(contract, observation)
    assert result["status"] == "reject"
    assert any(error.startswith("topic_type_mismatch:") for error in result["errors"])


def test_frame_mismatch_rejects_ingest() -> None:
    contract = make_contract()
    observation = passing_observation(contract)
    observation["samples"][contract.lidar_topic]["frame_id"] = "wrong_lidar"
    result = evaluate_observation(contract, observation)
    assert result["status"] == "reject"
    assert "lidar_frame_mismatch" in result["errors"]


def test_missing_optional_camera_info_and_tf_are_warnings_only() -> None:
    contract = make_contract()
    result = evaluate_observation(contract, passing_observation(contract))
    assert result["status"] == "pass"
    assert f"optional_topic_missing:{contract.camera_info_topic}" in result["warnings"]
    assert f"optional_topic_missing:{contract.tf_static_topic}" in result["warnings"]


def test_live_and_rosbag_observations_use_identical_checks() -> None:
    offline = make_contract(input_mode="rosbag")
    live = replace(offline, input_mode="live_topics", bag_path=None)
    offline_result = evaluate_observation(offline, passing_observation(offline))
    live_result = evaluate_observation(live, passing_observation(live))
    assert offline_result["status"] == live_result["status"] == "pass"
    assert offline_result["required_topic_contract"] == live_result["required_topic_contract"]


def test_rain_monitoring_requires_camera_but_not_lidar() -> None:
    contract = make_contract(operating_phase="rain_monitoring")
    observation = passing_observation(contract)
    observation["topic_types"].pop(contract.lidar_topic)
    observation["message_counts"].pop(contract.lidar_topic)
    observation["samples"].pop(contract.lidar_topic)
    result = evaluate_observation(contract, observation)
    assert result["status"] == "pass"
    assert contract.camera_topic in result["required_topic_contract"]
    assert contract.lidar_topic not in result["required_topic_contract"]
    assert f"optional_topic_missing:{contract.lidar_topic}" in result["warnings"]


def test_baseline_scan_requires_lidar_but_not_camera() -> None:
    contract = make_contract(operating_phase="baseline_scan")
    observation = passing_observation(contract)
    observation["topic_types"].pop(contract.camera_topic)
    observation["message_counts"].pop(contract.camera_topic)
    observation["samples"].pop(contract.camera_topic)
    result = evaluate_observation(contract, observation)
    assert result["status"] == "pass"
    assert contract.lidar_topic in result["required_topic_contract"]
    assert contract.camera_topic not in result["required_topic_contract"]
    assert f"optional_topic_missing:{contract.camera_topic}" in result["warnings"]


def test_two_stage_profiles_select_only_the_active_sensor() -> None:
    rain = make_contract(operating_phase="rain_monitoring")
    baseline = make_contract(operating_phase="baseline_scan")
    combined = make_contract(operating_phase="combined_validation")
    assert required_topic_contract(rain) == {
        rain.camera_topic: CAMERA_IMAGE_TYPE,
    }
    assert required_topic_contract(baseline) == {
        baseline.lidar_topic: POINT_CLOUD_TYPE,
    }
    assert required_topic_contract(combined) == {
        combined.camera_topic: CAMERA_IMAGE_TYPE,
        combined.lidar_topic: POINT_CLOUD_TYPE,
    }


def test_sensor_adapter_does_not_import_geometry_or_ground_truth_modules() -> None:
    source = Path("src/sensors/unified_sensor_input.py").read_text(encoding="utf-8")
    forbidden_imports = (
        "src.fusion",
        "src.hydrology",
        "src.evaluation",
        "ground_truth",
    )
    lowered = source.lower()
    assert "from src.fusion" not in source
    assert "from src.hydrology" not in source
    assert "from src.evaluation" not in source
    assert "import ground_truth" not in lowered
