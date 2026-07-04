#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline ROS2 rosbag readers used by water_agent_system."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


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
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read up to max_frames PointCloud2 messages and return an Nx3 xyz array."""
    bag = Path(bag_path).expanduser()
    max_frames = max(1, int(max_frames))
    rosbag2_py, deserialize_message, get_message, point_cloud2 = load_ros_modules()
    _ = rosbag2_py
    reader = open_sequential_reader(bag)
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

    print(f"{log_prefix} open bag: {bag}")
    print(f"{log_prefix} topic: {topic_name}, max_frames={max_frames}")
    while reader.has_next() and frame_count < max_frames:
        topic, data, timestamp = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, msg_type)
        points = pointcloud2_to_xyz(msg, point_cloud2)
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
