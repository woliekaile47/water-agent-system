"""ROS 2 publisher for deterministic Ground Truth, TF, and dry LiDAR points."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_msgs.msg import Float32
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def _quaternion_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _transform(
    stamp: Any,
    parent: str,
    child: str,
    pose: dict[str, float],
    pitch_key: str = "pitch_deg",
) -> TransformStamped:
    message = TransformStamped()
    message.header.stamp = stamp
    message.header.frame_id = parent
    message.child_frame_id = child
    message.transform.translation.x = float(pose["x_m"])
    message.transform.translation.y = float(pose["y_m"])
    message.transform.translation.z = float(pose["z_m"])
    quaternion = _quaternion_from_euler(
        math.radians(float(pose.get("roll_deg", 0.0))),
        math.radians(float(pose.get(pitch_key, 0.0))),
        math.radians(float(pose.get("yaw_deg", 0.0))),
    )
    message.transform.rotation.x = quaternion[0]
    message.transform.rotation.y = quaternion[1]
    message.transform.rotation.z = quaternion[2]
    message.transform.rotation.w = quaternion[3]
    return message


class GroundTruthPublisher(Node):
    def __init__(self) -> None:
        super().__init__("waterlogging_sim_ground_truth")
        self.declare_parameter("project_root", str(Path.cwd()))
        self.declare_parameter("manifest_path", "")
        project_root = Path(str(self.get_parameter("project_root").value)).expanduser().resolve()
        manifest_value = str(self.get_parameter("manifest_path").value)
        if not manifest_value:
            raise ValueError("manifest_path parameter is required")
        manifest_path = Path(manifest_value).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = project_root / manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.project_root = project_root
        self.sensors = self.manifest["config_snapshot"]["sensors"]
        self.scenario = self.manifest["config_snapshot"]["scenario"]
        self.topics = self.manifest["topic_names"]
        self.frames = self.manifest["coordinate_frames"]

        self.ground_dem = np.load(self._artifact("ground_dem_path")).astype(np.float32)
        self.dem_mask = np.load(self._artifact("dem_mask_gt_path")).astype(bool)
        self.depth_map = np.load(self._artifact("depth_map_gt_path")).astype(np.float32)
        from PIL import Image as PilImage

        self.camera_mask = np.asarray(PilImage.open(self._artifact("camera_mask_gt_path")).convert("L"))

        self.water_level_pub = self.create_publisher(Float32, self.topics["water_level_gt"], 10)
        self.water_mask_pub = self.create_publisher(Image, self.topics["water_mask_gt"], qos_profile_sensor_data)
        self.depth_map_pub = self.create_publisher(Image, self.topics["depth_map_gt"], qos_profile_sensor_data)
        self.lidar_pub = None
        self.lidar_cloud: PointCloud2 | None = None
        if bool(self.manifest["lidar_enabled"]):
            self.lidar_pub = self.create_publisher(
                PointCloud2, self.topics["lidar_points"], qos_profile_sensor_data
            )
            self.lidar_cloud = self._build_deterministic_lidar_cloud()
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.dynamic_broadcaster = TransformBroadcaster(self)
        self._publish_static_tf()
        self.create_timer(0.1, self._publish_ground_truth)
        self.create_timer(1.0, self._publish_dynamic_tf)
        self.get_logger().info(
            f"case={self.manifest['case_id']} lidar_enabled={self.manifest['lidar_enabled']} "
            f"camera_enabled={self.manifest['camera_enabled']} data_role=ground_truth"
        )

    def _artifact(self, key: str) -> Path:
        path = Path(str(self.manifest[key])).expanduser()
        return path if path.is_absolute() else self.project_root / path

    def _publish_static_tf(self) -> None:
        stamp = self.get_clock().now().to_msg()
        rig = self.sensors["sensor_rig"]["pose_map"]
        lidar = self.sensors["lidar"]["pose_on_rig"]
        camera = self.sensors["camera"]["pose_on_rig"]
        messages = [
            _transform(stamp, self.frames["map"], self.frames["sensor_mount"], rig),
            _transform(stamp, self.frames["sensor_mount"], self.frames["lidar"], lidar),
            _transform(
                stamp,
                self.frames["sensor_mount"],
                self.frames["camera"],
                camera,
                pitch_key="pitch_down_deg",
            ),
        ]
        optical_pose = {
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "roll_deg": -90.0,
            "pitch_deg": 0.0,
            "yaw_deg": -90.0,
        }
        messages.append(
            _transform(
                stamp,
                self.frames["camera"],
                self.frames["camera_optical"],
                optical_pose,
            )
        )
        self.static_broadcaster.sendTransform(messages)

    def _publish_dynamic_tf(self) -> None:
        identity = {
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
        }
        message = _transform(
            self.get_clock().now().to_msg(),
            self.frames["map"],
            self.frames["road"],
            identity,
        )
        self.dynamic_broadcaster.sendTransform(message)

    def _image_message(self, array: np.ndarray, encoding: str, frame_id: str) -> Image:
        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = frame_id
        message.height = int(array.shape[0])
        message.width = int(array.shape[1])
        message.encoding = encoding
        message.is_bigendian = False
        contiguous = np.ascontiguousarray(array)
        message.step = int(contiguous.strides[0])
        message.data = contiguous.tobytes()
        return message

    def _publish_ground_truth(self) -> None:
        level = Float32()
        level.data = (
            float(self.manifest["water_level_m"])
            if self.manifest["water_level_m"] is not None
            else float("nan")
        )
        self.water_level_pub.publish(level)
        self.water_mask_pub.publish(
            self._image_message(self.camera_mask.astype(np.uint8), "mono8", self.frames["camera_optical"])
        )
        self.depth_map_pub.publish(
            self._image_message(self.depth_map.astype(np.float32), "32FC1", self.frames["road"])
        )
        if self.lidar_pub is not None and self.lidar_cloud is not None:
            self.lidar_cloud.header.stamp = self.get_clock().now().to_msg()
            self.lidar_pub.publish(self.lidar_cloud)

    def _build_deterministic_lidar_cloud(self) -> PointCloud2:
        """Sample the configured road geometry; this is not a ray-traced LiDAR."""
        road = self.sensors["road"]
        lidar = self.sensors["lidar"]
        rig = self.sensors["sensor_rig"]["pose_map"]
        pose = lidar["pose_on_rig"]
        resolution = float(road["dem_resolution_m"])
        ny, nx = self.ground_dem.shape
        xs = -float(road["length_m"]) / 2.0 + resolution * (np.arange(nx) + 0.5)
        ys = -float(road["width_m"]) / 2.0 + resolution * (np.arange(ny) + 0.5)
        xx, yy = np.meshgrid(xs, ys)
        stride = max(1, int(lidar["pointcloud_stride_cells"]))
        world_points = np.column_stack(
            [
                xx[::stride, ::stride].ravel(),
                yy[::stride, ::stride].ravel(),
                self.ground_dem[::stride, ::stride].ravel(),
            ]
        ).astype(np.float64)
        origin = np.asarray(
            [
                float(rig["x_m"]) + float(pose["x_m"]),
                float(rig["y_m"]) + float(pose["y_m"]),
                float(rig["z_m"]) + float(pose["z_m"]),
            ],
            dtype=np.float64,
        )
        relative = world_points - origin
        yaw = math.radians(float(pose["yaw_deg"]))
        x_lidar = math.cos(yaw) * relative[:, 0] + math.sin(yaw) * relative[:, 1]
        y_lidar = -math.sin(yaw) * relative[:, 0] + math.cos(yaw) * relative[:, 1]
        z_lidar = relative[:, 2]
        ranges = np.sqrt(x_lidar**2 + y_lidar**2 + z_lidar**2)
        azimuth = np.degrees(np.arctan2(y_lidar, x_lidar))
        elevation = np.degrees(np.arctan2(z_lidar, np.hypot(x_lidar, y_lidar)))
        keep = (
            (ranges >= float(lidar["min_range_m"]))
            & (ranges <= float(lidar["max_range_m"]))
            & (azimuth >= float(lidar["horizontal_min_deg"]))
            & (azimuth <= float(lidar["horizontal_max_deg"]))
            & (elevation >= float(lidar["vertical_min_deg"]))
            & (elevation <= float(lidar["vertical_max_deg"]))
        )
        channel_count = int(lidar["vertical_channels"])
        vertical_span = float(lidar["vertical_max_deg"]) - float(lidar["vertical_min_deg"])
        rings = np.clip(
            np.rint(
                (elevation - float(lidar["vertical_min_deg"]))
                / vertical_span
                * (channel_count - 1)
            ),
            0,
            channel_count - 1,
        ).astype(np.uint16)
        points = [
            (float(x), float(y), float(z), 1.0, int(ring))
            for x, y, z, ring in zip(
                x_lidar[keep], y_lidar[keep], z_lidar[keep], rings[keep]
            )
        ]
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="ring", offset=16, datatype=PointField.UINT16, count=1),
        ]
        header = Header()
        header.frame_id = self.frames["lidar"]
        cloud = point_cloud2.create_cloud(header, fields, points)
        self.get_logger().info(
            f"deterministic_geometry_generator points={cloud.width} channels={channel_count}"
        )
        return cloud


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: GroundTruthPublisher | None = None
    try:
        node = GroundTruthPublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
            node = None
        if rclpy.ok():
            rclpy.try_shutdown()
