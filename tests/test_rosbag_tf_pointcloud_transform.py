"""Unit tests for TF-aware offline PointCloud2 coordinate handling."""

import math

import numpy as np
import pytest

from src.sensors.rosbag_reader import (
    normalize_frame_id,
    resolve_frame_transform,
    rigid_transform_matrix,
    transform_points_xyz,
)
from src.dem.build_ground_dem import interpolate_local_plane


def yaw_transform(x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
    half = math.radians(yaw_deg) / 2.0
    return rigid_transform_matrix((x, y, z), (0.0, 0.0, math.sin(half), math.cos(half)))


def test_normalize_frame_id_accepts_ros_leading_slash() -> None:
    assert normalize_frame_id(" /map ") == "map"


def test_rigid_transform_rotates_and_translates_points() -> None:
    matrix = yaw_transform(1.0, 2.0, 3.0, 90.0)
    result = transform_points_xyz(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), matrix)
    np.testing.assert_allclose(result, [[1.0, 3.0, 3.0]], atol=1e-6)


def test_resolve_frame_transform_composes_static_tf_chain() -> None:
    map_from_mount = yaw_transform(-7.0, -4.5, 0.0, 0.0)
    mount_from_lidar = yaw_transform(0.0, 0.0, 3.2, 32.7)
    resolved = resolve_frame_transform(
        {("map", "sensor_mount"): map_from_mount, ("sensor_mount", "lidar_link"): mount_from_lidar},
        "lidar_link",
        "map",
    )
    np.testing.assert_allclose(resolved, map_from_mount @ mount_from_lidar, atol=1e-9)


def test_resolve_frame_transform_supports_inverse_direction() -> None:
    map_from_lidar = yaw_transform(2.0, -1.0, 3.0, 15.0)
    resolved = resolve_frame_transform({("map", "lidar"): map_from_lidar}, "map", "lidar")
    np.testing.assert_allclose(resolved, np.linalg.inv(map_from_lidar), atol=1e-9)


def test_missing_tf_path_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="No static TF path"):
        resolve_frame_transform({("map", "camera"): np.eye(4)}, "lidar", "map")


def test_transform_points_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Nx3"):
        transform_points_xyz(np.ones((2, 4), dtype=np.float32), np.eye(4))


def test_local_plane_interpolation_preserves_planar_dem() -> None:
    yy, xx = np.indices((7, 9))
    expected = (0.02 * yy - 0.03 * xx + 1.2).astype(np.float32)
    valid = np.zeros(expected.shape, dtype=bool)
    valid[::2, ::2] = True
    sparse = np.where(valid, expected, np.nan).astype(np.float32)
    result, filled = interpolate_local_plane(sparse, valid, 20, neighbor_count=8)
    assert np.isfinite(result).all()
    assert np.all(filled[~valid])
    np.testing.assert_allclose(result, expected, atol=1e-5)
    np.testing.assert_array_equal(result[valid], sparse[valid])


def test_local_plane_interpolation_respects_maximum_distance() -> None:
    sparse = np.full((20, 20), np.nan, dtype=np.float32)
    sparse[0, 0] = 1.0
    sparse[0, 1] = 1.0
    sparse[1, 0] = 1.0
    valid = np.isfinite(sparse)
    result, filled = interpolate_local_plane(sparse, valid, 2, neighbor_count=3)
    assert np.isnan(result[-1, -1])
    assert not filled[-1, -1]
