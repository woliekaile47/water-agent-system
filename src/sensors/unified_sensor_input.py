#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transport-neutral sensor input contract for rosbag and live ROS2 topics.

This module deliberately stops at sensor-ingest readiness.  It never imports
water-mask, DEM fusion, depth inversion, warning, or Ground Truth code.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.sensors.rosbag_reader import (
    load_ros_modules,
    normalize_frame_id,
    open_sequential_reader,
)


CAMERA_IMAGE_TYPE = "sensor_msgs/msg/Image"
CAMERA_INFO_TYPE = "sensor_msgs/msg/CameraInfo"
POINT_CLOUD_TYPE = "sensor_msgs/msg/PointCloud2"
TF_TYPE = "tf2_msgs/msg/TFMessage"
SUPPORTED_INPUT_MODES = {"rosbag", "live_topics"}
SUPPORTED_OPERATING_PHASES = {
    "baseline_scan",
    "rain_monitoring",
    "combined_validation",
}

FORBIDDEN_PREDICTION_CONFIG_KEYS = {
    "ground_truth",
    "ground_truth_path",
    "ground_truth_mask",
    "water_level_gt",
    "depth_map_gt",
    "dem_water_mask_gt",
    "camera_water_mask_gt",
    "area_gt",
    "volume_gt",
    "nominal_depth_cm",
}


def _normalize_topic(topic: str) -> str:
    normalized = "/" + str(topic).strip().strip("/")
    if normalized == "/":
        raise ValueError("ROS topic must be non-empty")
    return normalized


def _find_forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            path = f"{prefix}.{key}" if prefix else key
            if (
                lowered in FORBIDDEN_PREDICTION_CONFIG_KEYS
                or lowered.endswith("_gt")
                or lowered.startswith("gt_")
            ):
                found.append(path)
            found.extend(_find_forbidden_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, f"{prefix}[{index}]"))
    return found


@dataclass(frozen=True)
class SensorInputContract:
    """Shared downstream-facing contract for offline and live transports."""

    schema_version: int
    input_mode: str
    operating_phase: str
    camera_topic: str
    lidar_topic: str
    camera_info_topic: str
    tf_static_topic: str
    expected_camera_frame: str
    expected_lidar_frame: str
    expected_image_width: int
    expected_image_height: int
    bag_path: str | None
    live_observation_timeout_seconds: float
    calibration_path: str | None
    calibration_required_for_geometry: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_sensor_input_contract(
    config_path: str | Path,
    *,
    input_mode: str | None = None,
    operating_phase: str | None = None,
    bag_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    live_timeout_seconds: float | None = None,
) -> SensorInputContract:
    """Load one contract while keeping transport selection as a pure override."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Sensor input config does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Sensor input config root must be a mapping")

    forbidden = _find_forbidden_keys(raw)
    if forbidden:
        raise ValueError(
            "Prediction-side sensor config contains forbidden Ground Truth fields: "
            + ", ".join(sorted(forbidden))
        )

    section = raw.get("sensor_input")
    if not isinstance(section, dict):
        raise ValueError("Missing sensor_input mapping")
    topics = section.get("topics")
    frames = section.get("expected_frames")
    image = section.get("expected_image")
    sources = section.get("sources")
    calibration = raw.get("calibration")
    safety = raw.get("safety")
    for name, value in (
        ("sensor_input.topics", topics),
        ("sensor_input.expected_frames", frames),
        ("sensor_input.expected_image", image),
        ("sensor_input.sources", sources),
        ("calibration", calibration),
        ("safety", safety),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"Missing or invalid {name} mapping")

    mode = str(input_mode or section.get("input_mode", "")).strip()
    if mode not in SUPPORTED_INPUT_MODES:
        raise ValueError(
            f"input_mode must be one of {sorted(SUPPORTED_INPUT_MODES)}, got {mode!r}"
        )
    phase = str(
        operating_phase or section.get("operating_phase", "")
    ).strip()
    if phase not in SUPPORTED_OPERATING_PHASES:
        raise ValueError(
            "operating_phase must be one of "
            f"{sorted(SUPPORTED_OPERATING_PHASES)}, got {phase!r}"
        )

    if bool(safety.get("allow_ground_truth", False)):
        raise ValueError("safety.allow_ground_truth must remain false")
    if bool(safety.get("allow_depth_inversion_without_calibration", False)):
        raise ValueError(
            "safety.allow_depth_inversion_without_calibration must remain false"
        )

    rosbag_source = sources.get("rosbag", {})
    live_source = sources.get("live_topics", {})
    if not isinstance(rosbag_source, dict) or not isinstance(live_source, dict):
        raise ValueError("sources.rosbag and sources.live_topics must be mappings")

    selected_bag = bag_path if bag_path is not None else rosbag_source.get("path")
    selected_calibration = (
        calibration_path
        if calibration_path is not None
        else calibration.get("path")
    )
    timeout = (
        float(live_timeout_seconds)
        if live_timeout_seconds is not None
        else float(live_source.get("observation_timeout_seconds", 5.0))
    )
    if timeout <= 0.0:
        raise ValueError("live observation timeout must be > 0")

    contract = SensorInputContract(
        schema_version=int(raw.get("schema_version", 1)),
        input_mode=mode,
        operating_phase=phase,
        camera_topic=_normalize_topic(topics["camera_image"]),
        lidar_topic=_normalize_topic(topics["lidar_points"]),
        camera_info_topic=_normalize_topic(topics["camera_info"]),
        tf_static_topic=_normalize_topic(topics["tf_static"]),
        expected_camera_frame=normalize_frame_id(frames["camera"]),
        expected_lidar_frame=normalize_frame_id(frames["lidar"]),
        expected_image_width=int(image["width"]),
        expected_image_height=int(image["height"]),
        bag_path=(
            str(Path(selected_bag).expanduser().resolve())
            if selected_bag
            else None
        ),
        live_observation_timeout_seconds=timeout,
        calibration_path=(
            str(Path(selected_calibration).expanduser().resolve())
            if selected_calibration
            else None
        ),
        calibration_required_for_geometry=bool(
            calibration.get("required_for_geometry", True)
        ),
    )
    if not contract.expected_camera_frame or not contract.expected_lidar_frame:
        raise ValueError("Expected camera and lidar frame ids must be non-empty")
    if contract.expected_image_width <= 0 or contract.expected_image_height <= 0:
        raise ValueError("Expected image dimensions must be positive")
    if contract.input_mode == "rosbag" and not contract.bag_path:
        raise ValueError("rosbag mode requires a bag path in config or --bag")
    return contract


def _finite_sequence(value: Any, length: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} numbers")
    converted = [float(item) for item in value]
    if not all(math.isfinite(item) for item in converted):
        raise ValueError(f"{field} contains NaN or Inf")
    return converted


def validate_calibration(
    contract: SensorInputContract,
) -> dict[str, Any]:
    """Validate an independent real calibration file without inventing defaults."""

    result: dict[str, Any] = {
        "status": "missing",
        "path": contract.calibration_path,
        "errors": [],
        "source": "independent_external_calibration",
        "ground_truth_used": False,
    }
    if not contract.calibration_path:
        result["errors"].append("calibration_path_not_configured")
        return result

    path = Path(contract.calibration_path)
    if not path.is_file():
        result["errors"].append("calibration_file_not_found")
        return result

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("calibration root must be a mapping")
        forbidden = _find_forbidden_keys(data)
        if forbidden:
            raise ValueError(
                "calibration contains forbidden Ground Truth fields: "
                + ", ".join(sorted(forbidden))
            )

        frames = data["frames"]
        camera = data["camera"]
        transforms = data["transforms"]
        if not all(isinstance(value, dict) for value in (frames, camera, transforms)):
            raise ValueError("frames, camera, and transforms must be mappings")

        if normalize_frame_id(frames["camera"]) != contract.expected_camera_frame:
            raise ValueError("calibration camera frame does not match sensor contract")
        if normalize_frame_id(frames["lidar"]) != contract.expected_lidar_frame:
            raise ValueError("calibration lidar frame does not match sensor contract")
        if not normalize_frame_id(frames["map"]):
            raise ValueError("calibration map frame is empty")
        if not normalize_frame_id(frames["camera_optical"]):
            raise ValueError("calibration camera optical frame is empty")

        if int(camera["image_width"]) != contract.expected_image_width:
            raise ValueError("calibration image width does not match sensor contract")
        if int(camera["image_height"]) != contract.expected_image_height:
            raise ValueError("calibration image height does not match sensor contract")
        _finite_sequence(camera["k"], 9, "camera.k")
        distortion = camera.get("d", [])
        if not isinstance(distortion, list):
            raise ValueError("camera.d must be a list")
        _finite_sequence(distortion, len(distortion), "camera.d")

        for transform_name in ("map_from_camera_optical", "map_from_lidar"):
            transform = transforms[transform_name]
            if not isinstance(transform, dict):
                raise ValueError(f"transforms.{transform_name} must be a mapping")
            _finite_sequence(
                transform["translation_xyz"],
                3,
                f"transforms.{transform_name}.translation_xyz",
            )
            quaternion = _finite_sequence(
                transform["quaternion_xyzw"],
                4,
                f"transforms.{transform_name}.quaternion_xyzw",
            )
            if math.sqrt(sum(item * item for item in quaternion)) <= 1e-12:
                raise ValueError(
                    f"transforms.{transform_name}.quaternion_xyzw has zero norm"
                )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        result["status"] = "invalid"
        result["errors"].append(str(exc))
        return result

    result["status"] = "valid"
    return result


def _message_sample(topic: str, message: Any, contract: SensorInputContract) -> dict[str, Any]:
    if topic == contract.camera_topic:
        return {
            "frame_id": normalize_frame_id(message.header.frame_id),
            "width": int(message.width),
            "height": int(message.height),
            "encoding": str(message.encoding),
            "step": int(message.step),
        }
    if topic == contract.lidar_topic:
        return {
            "frame_id": normalize_frame_id(message.header.frame_id),
            "width": int(message.width),
            "height": int(message.height),
            "point_step": int(message.point_step),
            "row_step": int(message.row_step),
            "fields": [str(field.name) for field in message.fields],
        }
    return {}


def required_topic_contract(
    contract: SensorInputContract,
) -> dict[str, str]:
    """Return only the sensors required by the active two-stage profile."""

    return {
        "baseline_scan": {
            contract.lidar_topic: POINT_CLOUD_TYPE,
        },
        "rain_monitoring": {
            contract.camera_topic: CAMERA_IMAGE_TYPE,
        },
        "combined_validation": {
            contract.camera_topic: CAMERA_IMAGE_TYPE,
            contract.lidar_topic: POINT_CLOUD_TYPE,
        },
    }[contract.operating_phase]


def inspect_rosbag_source(contract: SensorInputContract) -> dict[str, Any]:
    """Read one frozen bag without replaying it or starting device nodes."""

    if not contract.bag_path:
        raise ValueError("rosbag source requires contract.bag_path")
    bag = Path(contract.bag_path)
    _, deserialize_message, get_message, _ = load_ros_modules()
    reader = open_sequential_reader(bag)
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    message_types = {
        topic: get_message(msg_type)
        for topic, msg_type in topic_types.items()
        if topic in {contract.camera_topic, contract.lidar_topic}
    }
    counts = {topic: 0 for topic in topic_types}
    samples: dict[str, dict[str, Any]] = {}
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        counts[topic] = counts.get(topic, 0) + 1
        first_timestamp = timestamp if first_timestamp is None else first_timestamp
        last_timestamp = timestamp
        if topic in message_types and topic not in samples:
            message = deserialize_message(data, message_types[topic])
            samples[topic] = _message_sample(topic, message, contract)

    return {
        "transport": "rosbag",
        "source": str(bag),
        "topic_types": topic_types,
        "message_counts": counts,
        "samples": samples,
        "first_timestamp_ns": first_timestamp,
        "last_timestamp_ns": last_timestamp,
        "duration_seconds": (
            float(last_timestamp - first_timestamp) / 1e9
            if first_timestamp is not None and last_timestamp is not None
            else 0.0
        ),
    }


def inspect_live_topics_source(contract: SensorInputContract) -> dict[str, Any]:
    """Observe one camera and one LiDAR message without launching device drivers."""

    try:
        import rclpy
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import Image, PointCloud2
    except Exception as exc:
        raise RuntimeError(f"ROS2 live-topic dependencies are unavailable: {exc}") from exc

    rclpy.init(args=None)
    node = rclpy.create_node("water_agent_sensor_input_probe")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    samples: dict[str, dict[str, Any]] = {}
    counts = {contract.camera_topic: 0, contract.lidar_topic: 0}

    def camera_callback(message: Any) -> None:
        counts[contract.camera_topic] += 1
        samples.setdefault(
            contract.camera_topic,
            _message_sample(contract.camera_topic, message, contract),
        )

    def lidar_callback(message: Any) -> None:
        counts[contract.lidar_topic] += 1
        samples.setdefault(
            contract.lidar_topic,
            _message_sample(contract.lidar_topic, message, contract),
        )

    required_topics = required_topic_contract(contract)
    subscriptions = []
    if contract.camera_topic in required_topics:
        subscriptions.append(
            node.create_subscription(Image, contract.camera_topic, camera_callback, qos)
        )
    if contract.lidar_topic in required_topics:
        subscriptions.append(
            node.create_subscription(
                PointCloud2,
                contract.lidar_topic,
                lidar_callback,
                qos,
            )
        )
    _ = subscriptions
    deadline = time.monotonic() + contract.live_observation_timeout_seconds
    try:
        while time.monotonic() < deadline and not all(
            topic in samples for topic in required_topics
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        topic_types = {
            name: types[0] if len(types) == 1 else list(types)
            for name, types in node.get_topic_names_and_types()
        }
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    return {
        "transport": "live_topics",
        "source": "ros2_graph",
        "topic_types": topic_types,
        "message_counts": counts,
        "samples": samples,
        "observation_timeout_seconds": contract.live_observation_timeout_seconds,
    }


def evaluate_observation(
    contract: SensorInputContract,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the same topic, type, frame, and image checks to either transport."""

    topic_types = dict(observation.get("topic_types", {}))
    counts = dict(observation.get("message_counts", {}))
    samples = dict(observation.get("samples", {}))
    errors: list[str] = []
    warnings: list[str] = []

    required = required_topic_contract(contract)
    for topic, expected_type in required.items():
        observed_type = topic_types.get(topic)
        if observed_type is None:
            errors.append(f"required_topic_missing:{topic}")
        elif observed_type != expected_type:
            errors.append(
                f"topic_type_mismatch:{topic}:{observed_type}:{expected_type}"
            )
        if int(counts.get(topic, 0)) <= 0:
            errors.append(f"no_messages_observed:{topic}")

    camera_sample = samples.get(contract.camera_topic, {})
    lidar_sample = samples.get(contract.lidar_topic, {})
    if contract.camera_topic in required and camera_sample:
        if normalize_frame_id(camera_sample.get("frame_id", "")) != contract.expected_camera_frame:
            errors.append("camera_frame_mismatch")
        if int(camera_sample.get("width", 0)) != contract.expected_image_width:
            errors.append("camera_width_mismatch")
        if int(camera_sample.get("height", 0)) != contract.expected_image_height:
            errors.append("camera_height_mismatch")
    if contract.lidar_topic in required and lidar_sample:
        if normalize_frame_id(lidar_sample.get("frame_id", "")) != contract.expected_lidar_frame:
            errors.append("lidar_frame_mismatch")
        fields = set(lidar_sample.get("fields", []))
        if not {"x", "y", "z"}.issubset(fields):
            errors.append("lidar_xyz_fields_missing")

    optional = {
        contract.camera_info_topic: CAMERA_INFO_TYPE,
        contract.tf_static_topic: TF_TYPE,
    }
    if contract.camera_topic not in required:
        optional[contract.camera_topic] = CAMERA_IMAGE_TYPE
    if contract.lidar_topic not in required:
        optional[contract.lidar_topic] = POINT_CLOUD_TYPE
    for topic, expected_type in optional.items():
        observed_type = topic_types.get(topic)
        if observed_type is None:
            warnings.append(f"optional_topic_missing:{topic}")
        elif observed_type != expected_type:
            warnings.append(
                f"optional_topic_type_mismatch:{topic}:{observed_type}:{expected_type}"
            )

    return {
        "status": "pass" if not errors else "reject",
        "errors": errors,
        "warnings": warnings,
        "required_topic_contract": required,
        "optional_topic_contract": optional,
        "observed_topics": topic_types,
        "message_counts": counts,
        "samples": samples,
    }


def build_readiness_report(
    contract: SensorInputContract,
    observation: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    ingest = evaluate_observation(contract, observation)
    calibration_valid = calibration.get("status") == "valid"
    geometry_ready = ingest["status"] == "pass" and (
        calibration_valid or not contract.calibration_required_for_geometry
    )
    blocked_reasons: list[str] = []
    if ingest["status"] != "pass":
        blocked_reasons.append("sensor_data_ingest_rejected")
    if contract.calibration_required_for_geometry and not calibration_valid:
        blocked_reasons.append("independent_calibration_not_ready")

    return {
        "schema_version": contract.schema_version,
        "result_type": "sensor_input_readiness",
        "source_mode": contract.input_mode,
        "operating_phase": contract.operating_phase,
        "contract": contract.to_dict(),
        "data_ingest": ingest,
        "calibration": dict(calibration),
        "data_ingest_status": ingest["status"],
        "geometry_readiness": "ready" if geometry_ready else "blocked",
        "depth_inversion_allowed": bool(geometry_ready),
        "next_allowed_stage": "geometry" if geometry_ready else "data_ingest_only",
        "blocked_reasons": blocked_reasons,
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
        "real_deployment_contract_preserved": True,
        "note": (
            "Rosbag and live topics share this contract. Missing independent "
            "calibration blocks geometry and depth inversion without blocking "
            "safe sensor-ingest validation."
        ),
        "source_observation": dict(observation),
    }


def validate_sensor_input(
    config_path: str | Path,
    *,
    input_mode: str | None = None,
    operating_phase: str | None = None,
    bag_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    live_timeout_seconds: float | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    contract = load_sensor_input_contract(
        config_path,
        input_mode=input_mode,
        operating_phase=operating_phase,
        bag_path=bag_path,
        calibration_path=calibration_path,
        live_timeout_seconds=live_timeout_seconds,
    )
    observation = (
        inspect_rosbag_source(contract)
        if contract.input_mode == "rosbag"
        else inspect_live_topics_source(contract)
    )
    calibration = validate_calibration(contract)
    report = build_readiness_report(contract, observation, calibration)
    if output_path:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return report
