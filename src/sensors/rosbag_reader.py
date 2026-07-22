#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline ROS2 rosbag readers used by water_agent_system."""

from __future__ import annotations

import re
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


def normalize_frame_id(frame_id: str) -> str:
    """Normalize ROS frame ids for graph lookup without changing their meaning."""
    return str(frame_id).strip().lstrip("/")


def rigid_transform_matrix(
    translation_xyz: tuple[float, float, float],
    quaternion_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    """Return a 4x4 parent-from-child transform from a ROS translation/quaternion."""
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = float(np.sqrt(x * x + y * y + z * z + w * w))
    if norm <= 1e-12:
        raise ValueError("Transform quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return matrix


def transform_points_xyz(points: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    """Transform an Nx3 point array while preserving float32 output."""
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got shape={xyz.shape}")
    matrix = np.asarray(target_from_source, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("target_from_source must be a finite 4x4 matrix")
    transformed = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    return transformed.astype(np.float32)


def resolve_frame_transform(
    parent_from_child: dict[tuple[str, str], np.ndarray],
    source_frame: str,
    target_frame: str,
) -> np.ndarray:
    """Resolve target-from-source through a static TF tree/graph."""
    source = normalize_frame_id(source_frame)
    target = normalize_frame_id(target_frame)
    if not source or not target:
        raise ValueError("Source and target frame ids must be non-empty")
    if source == target:
        return np.eye(4, dtype=np.float64)

    adjacency: dict[str, list[tuple[str, np.ndarray]]] = {}
    for (parent_raw, child_raw), matrix_raw in parent_from_child.items():
        parent = normalize_frame_id(parent_raw)
        child = normalize_frame_id(child_raw)
        matrix = np.asarray(matrix_raw, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Static transform {parent}->{child} is not 4x4")
        adjacency.setdefault(child, []).append((parent, matrix))
        adjacency.setdefault(parent, []).append((child, np.linalg.inv(matrix)))

    queue: deque[tuple[str, np.ndarray]] = deque([(source, np.eye(4, dtype=np.float64))])
    visited = {source}
    while queue:
        current, current_from_source = queue.popleft()
        for neighbor, neighbor_from_current in adjacency.get(current, []):
            if neighbor in visited:
                continue
            neighbor_from_source = neighbor_from_current @ current_from_source
            if neighbor == target:
                return neighbor_from_source
            visited.add(neighbor)
            queue.append((neighbor, neighbor_from_source))
    available = sorted(adjacency)
    raise RuntimeError(
        f"No static TF path from {source} to {target}; available frames: {available}"
    )


def _fail(message: str) -> None:
    raise RuntimeError(message)


def load_ros_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from sensor_msgs_py import point_cloud2
    except Exception as exc:
        _fail(
            "缺少 ROS2 Python 依赖。请先 source /opt/ros/humble/setup.bash "
            "以及工作空间 install/setup.bash 后再运行；"
            f"原始错误: {exc}"
        )
    return rosbag2_py, deserialize_message, get_message, point_cloud2


def detect_storage_id(bag_path: Path) -> str:
    metadata = bag_path / "metadata.yaml"
    if not metadata.exists():
        return "sqlite3"
    text = metadata.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"storage_identifier:\s*([A-Za-z0-9_]+)", text)
    return match.group(1) if match else "sqlite3"


def open_sequential_reader(bag_path: Path):
    if not bag_path.exists():
        _fail(f"rosbag 路径不存在: {bag_path}")
    if not bag_path.is_dir():
        _fail(f"rosbag 路径应为目录: {bag_path}")

    rosbag2_py, _, _, _ = load_ros_modules()
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path),
        storage_id=detect_storage_id(bag_path),
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    return reader


def read_static_transforms(
    bag_path: str | Path,
    topic_name: str = "/tf_static",
) -> dict[tuple[str, str], np.ndarray]:
    """Read all TransformStamped records from a rosbag static-TF topic."""
    bag = Path(bag_path).expanduser()
    _, deserialize_message, get_message, _ = load_ros_modules()
    reader = open_sequential_reader(bag)
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topic_name not in topic_types:
        available = ", ".join(sorted(topic_types)) or "(none)"
        _fail(f"static TF topic does not exist: {topic_name}; available topics: {available}")
    if topic_types[topic_name] != "tf2_msgs/msg/TFMessage":
        _fail(f"topic {topic_name} is {topic_types[topic_name]}, not tf2_msgs/msg/TFMessage")

    msg_type = get_message(topic_types[topic_name])
    transforms: dict[tuple[str, str], np.ndarray] = {}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != topic_name:
            continue
        message = deserialize_message(data, msg_type)
        for stamped in message.transforms:
            parent = normalize_frame_id(stamped.header.frame_id)
            child = normalize_frame_id(stamped.child_frame_id)
            transform = stamped.transform
            transforms[(parent, child)] = rigid_transform_matrix(
                (
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                ),
                (
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ),
            )
    if not transforms:
        _fail(f"No transforms were read from {topic_name}")
    return transforms


def pointcloud2_to_xyz(msg: Any, point_cloud2: Any) -> np.ndarray:
    points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    if isinstance(points, np.ndarray):
        if points.dtype.names:
            arr = np.column_stack((points["x"], points["y"], points["z"]))
        else:
            arr = np.asarray(points)
            if arr.ndim == 1:
                arr = arr.reshape((-1, 3))
            else:
                arr = arr[:, :3]
    else:
        rows = list(points)
        if not rows:
            return np.empty((0, 3), dtype=np.float32)
        arr = np.asarray(rows, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape((-1, 3))
        else:
            arr = arr[:, :3]

    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr).all(axis=1)
    return arr[finite]


def read_pointcloud_xyz(
    bag_path: str | Path,
    topic_name: str,
    max_frames: int,
    log_prefix: str = "[rosbag_reader]",
    target_frame: str | None = None,
    static_tf_topic: str = "/tf_static",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read up to max_frames PointCloud2 messages and return an Nx3 xyz array."""
    bag = Path(bag_path).expanduser()
    max_frames = max(1, int(max_frames))
    rosbag2_py, deserialize_message, get_message, point_cloud2 = load_ros_modules()
    _ = rosbag2_py
    reader = open_sequential_reader(bag)
    static_transforms = read_static_transforms(bag, static_tf_topic) if target_frame else {}
    transform_cache: dict[str, np.ndarray] = {}
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topic_name not in topic_types:
        available = ", ".join(sorted(topic_types)) or "(none)"
        _fail(f"topic 不存在: {topic_name}; 当前 bag topics: {available}")
    if topic_types[topic_name] != "sensor_msgs/msg/PointCloud2":
        _fail(f"topic {topic_name} 类型不是 PointCloud2，而是 {topic_types[topic_name]}")

    msg_type = get_message(topic_types[topic_name])
    frame_count = 0
    point_count = 0
    chunks: list[np.ndarray] = []
    source_frames: set[str] = set()

    print(f"{log_prefix} open bag: {bag}")
    print(f"{log_prefix} topic: {topic_name}, max_frames={max_frames}")
    while reader.has_next() and frame_count < max_frames:
        topic, data, timestamp = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, msg_type)
        points = pointcloud2_to_xyz(msg, point_cloud2)
        source_frame = normalize_frame_id(msg.header.frame_id)
        source_frames.add(source_frame)
        if target_frame:
            if not source_frame:
                _fail(f"PointCloud2 on {topic_name} has an empty frame_id")
            if source_frame not in transform_cache:
                transform_cache[source_frame] = resolve_frame_transform(
                    static_transforms,
                    source_frame,
                    target_frame,
                )
            points = transform_points_xyz(points, transform_cache[source_frame])
        frame_count += 1
        point_count += int(points.shape[0])
        if points.size:
            chunks.append(points)
        print(
            f"{log_prefix} frame={frame_count:03d}, "
            f"timestamp={timestamp}, points={points.shape[0]}"
        )

    if frame_count == 0:
        _fail(f"没有从 topic {topic_name} 读取到任何 PointCloud2 帧")
    if not chunks:
        _fail("读取到 PointCloud2 帧，但没有有效 xyz 点")

    xyz = np.concatenate(chunks, axis=0)
    stats = {
        "bag_path": str(bag),
        "topic": topic_name,
        "frames_read": int(frame_count),
        "point_count": int(point_count),
        "x_min": float(np.min(xyz[:, 0])),
        "x_max": float(np.max(xyz[:, 0])),
        "y_min": float(np.min(xyz[:, 1])),
        "y_max": float(np.max(xyz[:, 1])),
        "z_min": float(np.min(xyz[:, 2])),
        "z_max": float(np.max(xyz[:, 2])),
        "source_frames": sorted(source_frames),
        "target_frame": normalize_frame_id(target_frame) if target_frame else None,
        "static_tf_topic": static_tf_topic if target_frame else None,
    }
    print(
        f"{log_prefix} done: frames={stats['frames_read']}, "
        f"points={stats['point_count']}, "
        f"x=[{stats['x_min']:.3f},{stats['x_max']:.3f}], "
        f"y=[{stats['y_min']:.3f},{stats['y_max']:.3f}], "
        f"z=[{stats['z_min']:.3f},{stats['z_max']:.3f}]"
    )
    sys.stdout.flush()
    return xyz, stats
