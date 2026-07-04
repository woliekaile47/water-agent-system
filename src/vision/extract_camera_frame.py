#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 helper: extract one camera Image frame from an offline ROS2 bag."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TOPIC = "/hik_camera/image_raw"


def load_ros_modules() -> tuple[Any, Any, Any]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception as exc:
        raise RuntimeError(
            "缺少 ROS2 Python 依赖。请先 source /opt/ros/humble/setup.bash "
            "以及工作空间 install/setup.bash 后再运行；"
            f"原始错误: {exc}"
        ) from exc
    return rosbag2_py, deserialize_message, get_message


def detect_storage_id(bag_path: Path) -> str:
    metadata = bag_path / "metadata.yaml"
    if not metadata.exists():
        return "sqlite3"
    text = metadata.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"storage_identifier:\s*([A-Za-z0-9_]+)", text)
    return match.group(1) if match else "sqlite3"


def open_reader(bag_path: Path):
    if not bag_path.exists():
        raise FileNotFoundError(f"rosbag 路径不存在: {bag_path}")
    if not bag_path.is_dir():
        raise ValueError(f"rosbag 路径应为目录: {bag_path}")

    rosbag2_py, _, _ = load_ros_modules()
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


def image_msg_to_array(msg: Any) -> np.ndarray:
    encoding = (msg.encoding or "").lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if height <= 0 or width <= 0 or step <= 0:
        raise ValueError(f"图像尺寸异常: width={width}, height={height}, step={step}")
    if raw.size < height * step:
        raise ValueError(f"图像数据长度不足: data={raw.size}, expected={height * step}")

    rows = raw[: height * step].reshape((height, step))
    if encoding in ("bgr8", "rgb8", "8uc3"):
        needed = width * 3
        if step < needed:
            raise ValueError(f"{encoding} 图像 step={step} 小于 needed={needed}")
        image = rows[:, :needed].reshape((height, width, 3)).copy()
        if encoding == "rgb8":
            image = image[:, :, ::-1].copy()
        return image
    if encoding in ("mono8", "8uc1"):
        if step < width:
            raise ValueError(f"{encoding} 图像 step={step} 小于 width={width}")
        return rows[:, :width].reshape((height, width)).copy()
    raise ValueError(f"暂不支持图像编码: {msg.encoding!r}，当前支持 bgr8/rgb8/mono8。")


def save_png(image: np.ndarray, output_path: Path) -> None:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"缺少 cv2，无法保存 PNG: {exc}") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"保存 PNG 失败: {output_path}")


def extract_camera_frame(
    bag_path: str | Path,
    output_path: str | Path,
    frame_index: int = 20,
    topic_name: str = DEFAULT_TOPIC,
) -> dict[str, Any]:
    if frame_index < 1:
        raise ValueError("--frame-index 从 1 开始计数，必须 >= 1")

    bag = Path(bag_path).expanduser()
    output = Path(output_path).expanduser()
    rosbag2_py, deserialize_message, get_message = load_ros_modules()
    _ = rosbag2_py
    reader = open_reader(bag)
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topic_name not in topic_types:
        available = ", ".join(sorted(topic_types)) or "(none)"
        raise RuntimeError(f"topic 不存在: {topic_name}; 当前 bag topics: {available}")
    if topic_types[topic_name] != "sensor_msgs/msg/Image":
        raise RuntimeError(f"topic {topic_name} 类型不是 sensor_msgs/msg/Image，而是 {topic_types[topic_name]}")

    msg_type = get_message("sensor_msgs/msg/Image")
    seen = 0
    print(f"[S3][extract_camera] bag={bag}")
    print(f"[S3][extract_camera] topic={topic_name}, target_frame_index={frame_index}")
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic != topic_name:
            continue
        seen += 1
        if seen == frame_index:
            msg = deserialize_message(data, msg_type)
            image = image_msg_to_array(msg)
            save_png(image, output)
            metadata = {
                "source_bag": str(bag),
                "topic": topic_name,
                "frame_index": int(seen),
                "timestamp": int(timestamp),
                "encoding": str(msg.encoding),
                "width": int(msg.width),
                "height": int(msg.height),
                "output": str(output),
            }
            print(
                "[S3][extract_camera] saved: "
                f"frame={seen}, encoding={msg.encoding}, size={msg.width}x{msg.height}, output={output}"
            )
            return metadata
        if seen % 10 == 0:
            print(f"[S3][extract_camera] read camera frames={seen}")

    raise RuntimeError(f"相机帧数量不足：只读取到 {seen} 帧，无法导出第 {frame_index} 帧")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one camera frame from an offline ROS2 bag.")
    parser.add_argument("--bag", required=True, help="ROS2 bag directory")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--frame-index", type=int, default=20, help="1-based camera frame index")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Camera Image topic")
    args = parser.parse_args()
    extract_camera_frame(args.bag, args.output, args.frame_index, args.topic)


if __name__ == "__main__":
    main()
