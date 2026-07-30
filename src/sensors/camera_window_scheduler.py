#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed 60-second Camera window scheduling for the two-stage pipeline.

The module only consumes RGB Camera messages and independent calibration
status. It never reads simulation Ground Truth, water levels, masks, depth,
area, or volume. A valid temporal window may be handed to the existing
temporal/SAM2 pipeline; geometry remains blocked until calibration is valid.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from PIL import Image

from src.sensors.rosbag_reader import (
    load_ros_modules,
    normalize_frame_id,
    open_sequential_reader,
)
from src.sensors.unified_sensor_input import (
    CAMERA_IMAGE_TYPE,
    SensorInputContract,
    load_sensor_input_contract,
    validate_calibration,
)


ERROR_CATALOG: dict[str, tuple[str, str]] = {
    "no_camera_frames": (
        "没有收到摄像头图像。",
        "检查摄像头节点、Camera topic、网络连接和 ROS2 环境。",
    ),
    "camera_frame_count_below_minimum": (
        "60 秒窗口内有效图像帧数不足。",
        "检查实际帧率、丢帧和录制时长；不要用重复帧补足。",
    ),
    "camera_window_span_below_minimum": (
        "摄像头数据覆盖时间不足 60 秒窗口的最低要求。",
        "继续采集，直到时间跨度达到配置要求。",
    ),
    "camera_timestamp_not_strictly_increasing": (
        "摄像头时间戳没有严格递增。",
        "检查设备时间、ROS time 和 rosbag 录制顺序。",
    ),
    "camera_frame_gap_above_limit": (
        "摄像头帧之间存在过长中断。",
        "检查网络丢包、RTSP 解码和摄像头进程稳定性。",
    ),
    "camera_frame_mismatch": (
        "Camera frame_id 与统一传感器契约不一致。",
        "检查海康节点 frame_id 和统一输入配置。",
    ),
    "camera_image_size_mismatch": (
        "摄像头图像尺寸与统一传感器契约不一致。",
        "确认使用 640×360 子码流，或同步更新经过验证的统一配置。",
    ),
    "camera_encoding_unsupported": (
        "摄像头图像编码暂不支持。",
        "使用 rgb8、bgr8、rgba8、bgra8 或 mono8。",
    ),
    "uniform_sampling_incomplete": (
        "无法从当前窗口形成完整的 41 帧均匀样本。",
        "检查窗口时长、帧数和时间戳连续性。",
    ),
    "uniform_sampling_not_unique": (
        "均匀采样出现重复源帧。",
        "提高输入帧率或降低允许的最大帧间隔，不要复制帧。",
    ),
    "independent_calibration_not_ready": (
        "独立真实相机标定尚未就绪，禁止继续 DEM 几何和水深反演。",
        "现场完成相机内参及 Camera-to-map 外参标定后再启用几何链路。",
    ),
}


def _problem(code: str, **details: Any) -> dict[str, Any]:
    message, action = ERROR_CATALOG[code]
    return {
        "code": code,
        "message": message,
        "recommended_action": action,
        "details": details,
    }


def _find_forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            path = f"{prefix}.{key}" if prefix else key
            if (
                (
                    "ground_truth" in lowered
                    and lowered != "allow_ground_truth"
                )
                or lowered.endswith("_gt")
                or lowered.startswith("gt_")
                or lowered in {
                    "nominal_depth_cm",
                    "water_level",
                    "depth_map",
                    "area",
                    "volume",
                }
            ):
                found.append(path)
            found.extend(_find_forbidden_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, f"{prefix}[{index}]"))
    return found


@dataclass(frozen=True)
class CameraWindowPolicy:
    schema_version: int
    window_duration_seconds: float
    sample_count: int
    minimum_source_frame_count: int
    minimum_coverage_ratio: float
    maximum_inter_frame_gap_seconds: float
    collection_timeout_seconds: float
    allowed_encodings: tuple[str, ...]

    @property
    def minimum_window_span_seconds(self) -> float:
        return self.window_duration_seconds * self.minimum_coverage_ratio

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_encodings"] = list(self.allowed_encodings)
        value["minimum_window_span_seconds"] = self.minimum_window_span_seconds
        return value


@dataclass
class CameraFrameRecord:
    timestamp_ns: int
    source_index: int
    width: int
    height: int
    frame_id: str
    encoding: str
    rgb: np.ndarray | None = field(default=None, repr=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "timestamp_ns": int(self.timestamp_ns),
            "source_index": int(self.source_index),
            "width": int(self.width),
            "height": int(self.height),
            "frame_id": self.frame_id,
            "encoding": self.encoding,
        }


def load_camera_window_policy(config_path: str | Path) -> CameraWindowPolicy:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Camera window config does not exist: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Camera window config root must be a mapping")
    forbidden = _find_forbidden_keys(document)
    if forbidden:
        raise ValueError(
            "Prediction-side Camera window config contains forbidden fields: "
            + ", ".join(sorted(forbidden))
        )
    section = document.get("camera_window")
    safety = document.get("safety")
    if not isinstance(section, dict) or not isinstance(safety, dict):
        raise ValueError("camera_window and safety mappings are required")
    if bool(safety.get("allow_ground_truth", False)):
        raise ValueError("safety.allow_ground_truth must remain false")
    if not bool(safety.get("require_independent_calibration_for_geometry", True)):
        raise ValueError(
            "safety.require_independent_calibration_for_geometry must remain true"
        )
    policy = CameraWindowPolicy(
        schema_version=int(document.get("schema_version", 1)),
        window_duration_seconds=float(section["duration_seconds"]),
        sample_count=int(section["sample_count"]),
        minimum_source_frame_count=int(section["minimum_source_frame_count"]),
        minimum_coverage_ratio=float(section["minimum_coverage_ratio"]),
        maximum_inter_frame_gap_seconds=float(
            section["maximum_inter_frame_gap_seconds"]
        ),
        collection_timeout_seconds=float(section["collection_timeout_seconds"]),
        allowed_encodings=tuple(
            str(item).lower() for item in section["allowed_encodings"]
        ),
    )
    if policy.window_duration_seconds <= 0.0:
        raise ValueError("camera_window.duration_seconds must be > 0")
    if policy.sample_count < 2:
        raise ValueError("camera_window.sample_count must be >= 2")
    if policy.minimum_source_frame_count < policy.sample_count:
        raise ValueError(
            "minimum_source_frame_count must be >= sample_count"
        )
    if not 0.0 < policy.minimum_coverage_ratio <= 1.0:
        raise ValueError("minimum_coverage_ratio must be in (0, 1]")
    if policy.maximum_inter_frame_gap_seconds <= 0.0:
        raise ValueError("maximum_inter_frame_gap_seconds must be > 0")
    if policy.collection_timeout_seconds < policy.window_duration_seconds:
        raise ValueError(
            "collection_timeout_seconds must be >= duration_seconds"
        )
    if not policy.allowed_encodings:
        raise ValueError("allowed_encodings must not be empty")
    return policy


def image_message_to_rgb(message: Any, allowed_encodings: tuple[str, ...]) -> np.ndarray:
    encoding = str(message.encoding).lower()
    if encoding not in allowed_encodings:
        raise ValueError(f"camera_encoding_unsupported:{encoding}")
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    channel_count = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }[encoding]
    row_bytes = width * channel_count
    if width <= 0 or height <= 0 or step < row_bytes:
        raise ValueError(
            f"invalid Camera image layout: width={width}, height={height}, step={step}"
        )
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    required = height * step
    if raw.size < required:
        raise ValueError(
            f"Camera image data is truncated: bytes={raw.size}, required={required}"
        )
    pixels = raw[:required].reshape(height, step)[:, :row_bytes]
    pixels = pixels.reshape(height, width, channel_count)
    if encoding == "bgr8":
        rgb = pixels[:, :, ::-1]
    elif encoding == "bgra8":
        rgb = pixels[:, :, [2, 1, 0]]
    elif encoding == "rgba8":
        rgb = pixels[:, :, :3]
    elif encoding == "mono8":
        rgb = np.repeat(pixels, 3, axis=2)
    else:
        rgb = pixels
    return np.ascontiguousarray(rgb, dtype=np.uint8)


class UniformCameraWindowBuffer:
    """Select 41 nearest frames online while retaining only selected payloads."""

    def __init__(
        self,
        policy: CameraWindowPolicy,
        contract: SensorInputContract,
    ) -> None:
        self.policy = policy
        self.contract = contract
        self.source_frame_count = 0
        self.first_timestamp_ns: int | None = None
        self.last_timestamp_ns: int | None = None
        self.maximum_gap_ns = 0
        self._target_timestamps_ns: list[int] = []
        self._target_cursor = 0
        self._previous: CameraFrameRecord | None = None
        self.selected: list[CameraFrameRecord] = []
        self.problems: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

    @property
    def complete(self) -> bool:
        return self._target_cursor >= self.policy.sample_count

    def _record_problem(self, code: str, **details: Any) -> None:
        if not any(item["code"] == code for item in self.problems):
            self.problems.append(_problem(code, **details))

    def _validate_record(self, record: CameraFrameRecord) -> None:
        if normalize_frame_id(record.frame_id) != self.contract.expected_camera_frame:
            self._record_problem(
                "camera_frame_mismatch",
                observed=record.frame_id,
                expected=self.contract.expected_camera_frame,
            )
        if (
            int(record.width) != self.contract.expected_image_width
            or int(record.height) != self.contract.expected_image_height
        ):
            self._record_problem(
                "camera_image_size_mismatch",
                observed=[int(record.width), int(record.height)],
                expected=[
                    self.contract.expected_image_width,
                    self.contract.expected_image_height,
                ],
            )
        if record.encoding.lower() not in self.policy.allowed_encodings:
            self._record_problem(
                "camera_encoding_unsupported",
                observed=record.encoding,
                allowed=list(self.policy.allowed_encodings),
            )

    def add(self, record: CameraFrameRecord) -> None:
        if self.complete:
            return
        record.timestamp_ns = int(record.timestamp_ns)
        record.source_index = int(record.source_index)
        self._validate_record(record)
        if self.first_timestamp_ns is None:
            self.first_timestamp_ns = record.timestamp_ns
            duration_ns = int(round(self.policy.window_duration_seconds * 1e9))
            self._target_timestamps_ns = [
                int(round(value))
                for value in np.linspace(
                    self.first_timestamp_ns,
                    self.first_timestamp_ns + duration_ns,
                    self.policy.sample_count,
                )
            ]
        elif self.last_timestamp_ns is not None:
            gap_ns = record.timestamp_ns - self.last_timestamp_ns
            if gap_ns <= 0:
                self._record_problem(
                    "camera_timestamp_not_strictly_increasing",
                    previous_timestamp_ns=self.last_timestamp_ns,
                    current_timestamp_ns=record.timestamp_ns,
                )
                return
            self.maximum_gap_ns = max(self.maximum_gap_ns, gap_ns)

        self.source_frame_count += 1
        current = record
        while (
            self._target_cursor < len(self._target_timestamps_ns)
            and self._target_timestamps_ns[self._target_cursor]
            <= current.timestamp_ns
        ):
            target = self._target_timestamps_ns[self._target_cursor]
            candidates = [current]
            if self._previous is not None:
                candidates.append(self._previous)
            already_selected = {item.source_index for item in self.selected}
            available = [
                item for item in candidates if item.source_index not in already_selected
            ]
            if available:
                chosen = min(
                    available,
                    key=lambda item: (
                        abs(item.timestamp_ns - target),
                        item.timestamp_ns,
                        item.source_index,
                    ),
                )
                self.selected.append(chosen)
            self._target_cursor += 1
        self._previous = current
        self.last_timestamp_ns = current.timestamp_ns

    def finalize(
        self,
        calibration: Mapping[str, Any],
        *,
        source: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.source_frame_count == 0:
            self._record_problem("no_camera_frames")
        if self.source_frame_count < self.policy.minimum_source_frame_count:
            self._record_problem(
                "camera_frame_count_below_minimum",
                observed=self.source_frame_count,
                required=self.policy.minimum_source_frame_count,
            )
        span_seconds = (
            float(self.last_timestamp_ns - self.first_timestamp_ns) / 1e9
            if self.first_timestamp_ns is not None
            and self.last_timestamp_ns is not None
            else 0.0
        )
        if span_seconds + 1e-9 < self.policy.minimum_window_span_seconds:
            self._record_problem(
                "camera_window_span_below_minimum",
                observed_seconds=span_seconds,
                required_seconds=self.policy.minimum_window_span_seconds,
            )
        maximum_gap_seconds = float(self.maximum_gap_ns) / 1e9
        if maximum_gap_seconds > self.policy.maximum_inter_frame_gap_seconds:
            self._record_problem(
                "camera_frame_gap_above_limit",
                observed_seconds=maximum_gap_seconds,
                maximum_seconds=self.policy.maximum_inter_frame_gap_seconds,
            )
        if len(self.selected) != self.policy.sample_count:
            self._record_problem(
                "uniform_sampling_incomplete",
                observed=len(self.selected),
                required=self.policy.sample_count,
            )
        selected_indices = [item.source_index for item in self.selected]
        if len(set(selected_indices)) != len(selected_indices):
            self._record_problem(
                "uniform_sampling_not_unique",
                selected_source_indices=selected_indices,
            )

        window_ready = not self.problems
        calibration_ready = calibration.get("status") == "valid"
        geometry_problems: list[dict[str, Any]] = []
        if not calibration_ready:
            geometry_problems.append(
                _problem(
                    "independent_calibration_not_ready",
                    calibration_status=calibration.get("status"),
                    calibration_errors=list(calibration.get("errors", [])),
                )
            )
        temporal_allowed = bool(window_ready)
        geometry_allowed = bool(window_ready and calibration_ready)
        sample_span_seconds = (
            float(
                self.selected[-1].timestamp_ns - self.selected[0].timestamp_ns
            )
            / 1e9
            if len(self.selected) >= 2
            else 0.0
        )
        effective_rate = (
            float(len(self.selected) - 1) / sample_span_seconds
            if sample_span_seconds > 0.0
            else 0.0
        )
        return {
            "schema_version": self.policy.schema_version,
            "result_type": "camera_window_schedule",
            "window_status": "pass" if window_ready else "reject",
            "pipeline_status": "ready" if geometry_allowed else "blocked",
            "temporal_sam2_handoff_allowed": temporal_allowed,
            "geometry_handoff_status": "ready" if geometry_allowed else "blocked",
            "depth_inversion_allowed": geometry_allowed,
            "next_allowed_stage": (
                "temporal_sam2_then_geometry"
                if geometry_allowed
                else "temporal_sam2_only"
                if temporal_allowed
                else "none"
            ),
            "window_problems": list(self.problems),
            "geometry_problems": geometry_problems,
            "warnings": list(self.warnings),
            "policy": self.policy.to_dict(),
            "source_frame_count": int(self.source_frame_count),
            "first_timestamp_ns": self.first_timestamp_ns,
            "last_timestamp_ns": self.last_timestamp_ns,
            "observed_span_seconds": span_seconds,
            "maximum_inter_frame_gap_seconds": maximum_gap_seconds,
            "sample_count": len(self.selected),
            "sample_span_seconds": sample_span_seconds,
            "effective_sample_rate_hz": effective_rate,
            "sampled_frames": [item.metadata() for item in self.selected],
            "calibration": dict(calibration),
            "source": dict(source or {}),
            "ground_truth_used": False,
            "authoritative": False,
            "eligible_for_downstream": False,
            "formal_water_depth_generated": False,
            "note": (
                "A passing window may enter the existing temporal/SAM2 chain. "
                "Independent calibration is additionally required before DEM "
                "geometry or water-depth inversion."
            ),
        }


def _header_timestamp_ns(message: Any, fallback_ns: int) -> tuple[int, bool]:
    stamp = message.header.stamp
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if value > 0:
        return value, False
    return int(fallback_ns), True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_sampled_window(
    output_dir: str | Path,
    selected: list[CameraFrameRecord],
    report: dict[str, Any],
) -> dict[str, Any]:
    if report["window_status"] != "pass":
        return report
    root = Path(output_dir).expanduser().resolve()
    frames_dir = root / "frames"
    if frames_dir.exists() and any(frames_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing Camera window frames: {frames_dir}"
        )
    frames_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    for output_index, record in enumerate(selected):
        if record.rgb is None:
            raise ValueError("Selected Camera frame payload is unavailable")
        expected_shape = (int(record.height), int(record.width), 3)
        if record.rgb.shape != expected_shape:
            raise ValueError(
                f"Selected Camera RGB shape is {record.rgb.shape}, "
                f"expected {expected_shape}"
            )
        path = frames_dir / f"frame_{output_index:06d}.png"
        Image.fromarray(record.rgb, mode="RGB").save(path)
        item = record.metadata()
        item.update(
            {
                "output_index": output_index,
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
        exported.append(item)
    reference_index = len(exported) // 2
    report["export"] = {
        "frames_dir": str(frames_dir),
        "exported_frame_count": len(exported),
        "reference_frame_index": reference_index,
        "reference_frame_path": exported[reference_index]["path"],
        "frames": exported,
        "handoff_contract": (
            "contiguous frame_000000.png sequence compatible with the existing "
            "temporal/SAM2 prompt pipeline"
        ),
    }
    manifest = root / "camera_window_manifest.json"
    manifest.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report["export"]["manifest_path"] = str(manifest)
    return report


def collect_rosbag_camera_window(
    contract: SensorInputContract,
    policy: CameraWindowPolicy,
    calibration: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    if not contract.bag_path:
        raise ValueError("rosbag Camera window scheduling requires a bag path")
    bag = Path(contract.bag_path)
    _, deserialize_message, get_message, _ = load_ros_modules()
    reader = open_sequential_reader(bag)
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    observed_type = topic_types.get(contract.camera_topic)
    if observed_type != CAMERA_IMAGE_TYPE:
        raise RuntimeError(
            f"Camera topic {contract.camera_topic} type is {observed_type!r}, "
            f"expected {CAMERA_IMAGE_TYPE}"
        )
    message_type = get_message(observed_type)
    buffer = UniformCameraWindowBuffer(policy, contract)
    source_index = 0
    while reader.has_next() and not buffer.complete:
        topic, data, bag_timestamp_ns = reader.read_next()
        if topic != contract.camera_topic:
            continue
        message = deserialize_message(data, message_type)
        timestamp_ns, used_fallback = _header_timestamp_ns(
            message,
            bag_timestamp_ns,
        )
        if used_fallback and not any(
            item.get("code") == "camera_header_timestamp_zero_using_bag_time"
            for item in buffer.warnings
        ):
            buffer.warnings.append(
                {
                    "code": "camera_header_timestamp_zero_using_bag_time",
                    "message": "Camera header 时间戳为 0，离线检查改用 rosbag 时间。",
                }
            )
        try:
            rgb = image_message_to_rgb(message, policy.allowed_encodings)
        except ValueError as exc:
            if str(exc).startswith("camera_encoding_unsupported:"):
                buffer._record_problem(
                    "camera_encoding_unsupported",
                    observed=str(message.encoding),
                    allowed=list(policy.allowed_encodings),
                )
                rgb = None
            else:
                raise
        buffer.add(
            CameraFrameRecord(
                timestamp_ns=timestamp_ns,
                source_index=source_index,
                width=int(message.width),
                height=int(message.height),
                frame_id=normalize_frame_id(message.header.frame_id),
                encoding=str(message.encoding).lower(),
                rgb=rgb,
            )
        )
        source_index += 1
    report = buffer.finalize(
        calibration,
        source={
            "mode": "rosbag",
            "bag_path": str(bag.resolve()),
            "camera_topic": contract.camera_topic,
            "ground_truth_read": False,
        },
    )
    return export_sampled_window(output_dir, buffer.selected, report)


def collect_live_camera_window(
    contract: SensorInputContract,
    policy: CameraWindowPolicy,
    calibration: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Observe an already-published Camera topic; never launches the device."""

    try:
        import rclpy
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import Image as RosImage
    except Exception as exc:
        raise RuntimeError(
            "ROS2 live Camera dependencies are unavailable. "
            "Source /opt/ros/humble/setup.bash first. "
            f"Original error: {exc}"
        ) from exc

    rclpy.init(args=None)
    node = rclpy.create_node("water_agent_camera_window_scheduler")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    buffer = UniformCameraWindowBuffer(policy, contract)

    def callback(message: Any) -> None:
        timestamp_ns, used_fallback = _header_timestamp_ns(
            message,
            node.get_clock().now().nanoseconds,
        )
        if used_fallback and not any(
            item.get("code") == "camera_header_timestamp_zero_using_node_clock"
            for item in buffer.warnings
        ):
            buffer.warnings.append(
                {
                    "code": "camera_header_timestamp_zero_using_node_clock",
                    "message": "Camera header 时间戳为 0，在线检查改用节点时钟。",
                }
            )
        try:
            rgb = image_message_to_rgb(message, policy.allowed_encodings)
        except ValueError as exc:
            if str(exc).startswith("camera_encoding_unsupported:"):
                buffer._record_problem(
                    "camera_encoding_unsupported",
                    observed=str(message.encoding),
                    allowed=list(policy.allowed_encodings),
                )
                return
            raise
        buffer.add(
            CameraFrameRecord(
                timestamp_ns=timestamp_ns,
                source_index=buffer.source_frame_count,
                width=int(message.width),
                height=int(message.height),
                frame_id=normalize_frame_id(message.header.frame_id),
                encoding=str(message.encoding).lower(),
                rgb=rgb,
            )
        )

    subscription = node.create_subscription(
        RosImage,
        contract.camera_topic,
        callback,
        qos,
    )
    _ = subscription
    deadline = time.monotonic() + policy.collection_timeout_seconds
    try:
        while time.monotonic() < deadline and not buffer.complete:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    report = buffer.finalize(
        calibration,
        source={
            "mode": "live_topics",
            "camera_topic": contract.camera_topic,
            "collection_timeout_seconds": policy.collection_timeout_seconds,
            "device_node_launched_by_scheduler": False,
            "ground_truth_read": False,
        },
    )
    return export_sampled_window(output_dir, buffer.selected, report)


def collect_frame_directory_window(
    contract: SensorInputContract,
    policy: CameraWindowPolicy,
    calibration: Mapping[str, Any],
    frames_dir: str | Path,
    source_fps: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Schedule a GT-isolated simulation frames/ directory.

    Only ``frame_*.png`` files inside the supplied directory are opened. Parent
    manifests, labels, Ground Truth, case names, and evaluation outputs are not
    inspected.
    """

    directory = Path(frames_dir).expanduser().resolve()
    if not directory.is_dir() or directory.name != "frames":
        raise ValueError(
            f"frame_directory input must be an existing directory named frames: "
            f"{directory}"
        )
    fps = float(source_fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("source_fps must be a finite number > 0")
    paths = sorted(directory.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.png images in {directory}")

    buffer = UniformCameraWindowBuffer(policy, contract)
    for source_index, path in enumerate(paths):
        expected_name = f"frame_{source_index:06d}.png"
        if path.name != expected_name:
            raise ValueError(
                f"Non-contiguous simulation frame numbering: expected "
                f"{expected_name}, got {path.name}"
            )
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        buffer.add(
            CameraFrameRecord(
                timestamp_ns=int(round(source_index / fps * 1e9)),
                source_index=source_index,
                width=int(rgb.shape[1]),
                height=int(rgb.shape[0]),
                frame_id=contract.expected_camera_frame,
                encoding="rgb8",
                rgb=rgb,
            )
        )
        if buffer.complete:
            break
    report = buffer.finalize(
        calibration,
        source={
            "mode": "frame_directory",
            "frames_dir": str(directory),
            "source_fps": fps,
            "parent_metadata_read": False,
            "ground_truth_read": False,
            "simulation_only": True,
        },
    )
    return export_sampled_window(output_dir, buffer.selected, report)


def run_camera_window_scheduler(
    *,
    window_config_path: str | Path,
    sensor_config_path: str | Path,
    input_mode: str,
    output_dir: str | Path,
    bag_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    frames_dir: str | Path | None = None,
    source_fps: float | None = None,
) -> dict[str, Any]:
    policy = load_camera_window_policy(window_config_path)
    if input_mode not in {"rosbag", "live_topics", "frame_directory"}:
        raise ValueError(
            "input_mode must be rosbag, live_topics, or frame_directory"
        )
    contract = load_sensor_input_contract(
        sensor_config_path,
        input_mode=(
            "live_topics" if input_mode == "frame_directory" else input_mode
        ),
        operating_phase="rain_monitoring",
        bag_path=bag_path,
        calibration_path=calibration_path,
    )
    calibration = validate_calibration(contract)
    if input_mode == "rosbag":
        report = collect_rosbag_camera_window(
            contract,
            policy,
            calibration,
            output_dir,
        )
    elif input_mode == "live_topics":
        report = collect_live_camera_window(
            contract,
            policy,
            calibration,
            output_dir,
        )
    else:
        if frames_dir is None or source_fps is None:
            raise ValueError(
                "frame_directory mode requires frames_dir and source_fps"
            )
        report = collect_frame_directory_window(
            contract,
            policy,
            calibration,
            frames_dir,
            source_fps,
            output_dir,
        )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    readiness = output / "camera_window_readiness.json"
    readiness.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report["readiness_path"] = str(readiness)
    return report
