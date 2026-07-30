"""Tests for deterministic, fail-closed one-minute Camera scheduling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.sensors.camera_window_scheduler import (
    CameraFrameRecord,
    CameraWindowPolicy,
    UniformCameraWindowBuffer,
    collect_frame_directory_window,
    export_sampled_window,
    load_camera_window_policy,
)
from src.sensors.unified_sensor_input import SensorInputContract


def policy(**overrides) -> CameraWindowPolicy:
    values = {
        "schema_version": 1,
        "window_duration_seconds": 60.0,
        "sample_count": 41,
        "minimum_source_frame_count": 41,
        "minimum_coverage_ratio": 1.0,
        "maximum_inter_frame_gap_seconds": 2.0,
        "collection_timeout_seconds": 75.0,
        "allowed_encodings": ("rgb8", "bgr8"),
    }
    values.update(overrides)
    return CameraWindowPolicy(**values)


def contract() -> SensorInputContract:
    return SensorInputContract(
        schema_version=1,
        input_mode="live_topics",
        operating_phase="rain_monitoring",
        camera_topic="/hik_camera/image_raw",
        lidar_topic="/cx/lslidar_point_cloud",
        camera_info_topic="/hik_camera/camera_info",
        tf_static_topic="/tf_static",
        expected_camera_frame="hik_camera_link",
        expected_lidar_frame="laser_link",
        expected_image_width=640,
        expected_image_height=360,
        bag_path=None,
        live_observation_timeout_seconds=5.0,
        calibration_path=None,
        calibration_required_for_geometry=True,
    )


def record(index: int, seconds: float, *, rgb: bool = False) -> CameraFrameRecord:
    return CameraFrameRecord(
        timestamp_ns=int(round(seconds * 1e9)),
        source_index=index,
        width=640,
        height=360,
        frame_id="hik_camera_link",
        encoding="rgb8",
        rgb=(
            np.full((360, 640, 3), index % 255, dtype=np.uint8)
            if rgb
            else None
        ),
    )


def full_buffer(*, with_rgb: bool = False) -> UniformCameraWindowBuffer:
    buffer = UniformCameraWindowBuffer(policy(), contract())
    for index in range(601):
        buffer.add(record(index, index / 10.0, rgb=with_rgb))
        if buffer.complete:
            break
    return buffer


def test_exact_one_minute_window_selects_41_unique_uniform_frames() -> None:
    buffer = full_buffer()
    report = buffer.finalize({"status": "valid", "errors": []})
    assert report["window_status"] == "pass"
    assert report["sample_count"] == 41
    indices = [item["source_index"] for item in report["sampled_frames"]]
    assert indices == list(range(0, 601, 15))
    assert len(set(indices)) == 41
    assert report["effective_sample_rate_hz"] == pytest.approx(2.0 / 3.0)


def test_same_input_produces_the_same_schedule() -> None:
    left = full_buffer().finalize({"status": "valid", "errors": []})
    right = full_buffer().finalize({"status": "valid", "errors": []})
    assert left["sampled_frames"] == right["sampled_frames"]


def test_short_bag_like_sequence_reports_time_span_error_in_plain_language() -> None:
    buffer = UniformCameraWindowBuffer(policy(), contract())
    for index in range(156):
        buffer.add(record(index, index / 10.0))
    report = buffer.finalize({"status": "missing", "errors": []})
    problems = {item["code"]: item for item in report["window_problems"]}
    assert report["window_status"] == "reject"
    assert "camera_window_span_below_minimum" in problems
    assert "覆盖时间不足" in problems["camera_window_span_below_minimum"]["message"]
    assert problems["camera_window_span_below_minimum"]["recommended_action"]
    assert report["formal_water_depth_generated"] is False


def test_too_few_frames_is_rejected_without_repeating_frames() -> None:
    buffer = UniformCameraWindowBuffer(policy(), contract())
    for index in range(20):
        buffer.add(record(index, index * (60.0 / 19.0)))
    report = buffer.finalize({"status": "valid", "errors": []})
    codes = {item["code"] for item in report["window_problems"]}
    assert "camera_frame_count_below_minimum" in codes
    assert "uniform_sampling_incomplete" in codes
    assert report["temporal_sam2_handoff_allowed"] is False


def test_long_camera_interruption_is_rejected() -> None:
    buffer = UniformCameraWindowBuffer(policy(), contract())
    times = list(np.arange(0.0, 20.1, 0.1)) + list(np.arange(25.0, 60.1, 0.1))
    for index, timestamp in enumerate(times):
        buffer.add(record(index, float(timestamp)))
        if buffer.complete:
            break
    report = buffer.finalize({"status": "valid", "errors": []})
    assert any(
        item["code"] == "camera_frame_gap_above_limit"
        for item in report["window_problems"]
    )


def test_missing_calibration_allows_temporal_but_blocks_depth() -> None:
    report = full_buffer().finalize(
        {"status": "missing", "errors": ["calibration_path_not_configured"]}
    )
    assert report["window_status"] == "pass"
    assert report["temporal_sam2_handoff_allowed"] is True
    assert report["geometry_handoff_status"] == "blocked"
    assert report["depth_inversion_allowed"] is False
    assert report["next_allowed_stage"] == "temporal_sam2_only"
    assert report["geometry_problems"][0]["code"] == (
        "independent_calibration_not_ready"
    )


def test_valid_calibration_allows_geometry_handoff_but_is_not_authoritative() -> None:
    report = full_buffer().finalize({"status": "valid", "errors": []})
    assert report["pipeline_status"] == "ready"
    assert report["geometry_handoff_status"] == "ready"
    assert report["depth_inversion_allowed"] is True
    assert report["authoritative"] is False
    assert report["eligible_for_downstream"] is False
    assert report["ground_truth_used"] is False


def test_frame_or_image_contract_mismatch_rejects_window() -> None:
    buffer = UniformCameraWindowBuffer(policy(), contract())
    bad = record(0, 0.0)
    bad.frame_id = "wrong_frame"
    bad.width = 1280
    buffer.add(bad)
    for index in range(1, 601):
        buffer.add(record(index, index / 10.0))
        if buffer.complete:
            break
    codes = {
        item["code"]
        for item in buffer.finalize({"status": "valid", "errors": []})[
            "window_problems"
        ]
    }
    assert "camera_frame_mismatch" in codes
    assert "camera_image_size_mismatch" in codes


def test_export_is_contiguous_and_compatible_with_existing_frames_loader(
    tmp_path: Path,
) -> None:
    buffer = full_buffer(with_rgb=True)
    report = buffer.finalize({"status": "valid", "errors": []})
    exported = export_sampled_window(tmp_path / "window", buffer.selected, report)
    frames = sorted((tmp_path / "window" / "frames").glob("frame_*.png"))
    assert len(frames) == 41
    assert frames[0].name == "frame_000000.png"
    assert frames[-1].name == "frame_000040.png"
    assert exported["export"]["reference_frame_index"] == 20
    manifest = json.loads(
        (tmp_path / "window" / "camera_window_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["ground_truth_used"] is False
    assert manifest["sample_count"] == 41


def test_prediction_side_config_rejects_ground_truth_fields(tmp_path: Path) -> None:
    config = {
        "schema_version": 1,
        "camera_window": {
            "duration_seconds": 60.0,
            "sample_count": 41,
            "minimum_source_frame_count": 41,
            "minimum_coverage_ratio": 1.0,
            "maximum_inter_frame_gap_seconds": 2.0,
            "collection_timeout_seconds": 75.0,
            "allowed_encodings": ["rgb8"],
            "water_level_gt": -0.1,
        },
        "safety": {
            "allow_ground_truth": False,
            "require_independent_calibration_for_geometry": True,
        },
    }
    path = tmp_path / "forbidden.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        load_camera_window_policy(path)


def test_repository_policy_loads_with_ground_truth_disabled() -> None:
    loaded = load_camera_window_policy("configs/camera_window_scheduler.yaml")
    assert loaded.window_duration_seconds == 60.0
    assert loaded.sample_count == 41
    assert loaded.minimum_coverage_ratio == 1.0


def test_frame_directory_mode_reads_only_contiguous_rgb_frames(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sequence" / "frames"
    source.mkdir(parents=True)
    for index in range(61):
        from PIL import Image

        Image.fromarray(
            np.full((360, 640, 3), index, dtype=np.uint8),
            mode="RGB",
        ).save(source / f"frame_{index:06d}.png")
    short_policy = policy(
        window_duration_seconds=6.0,
        sample_count=5,
        minimum_source_frame_count=5,
        collection_timeout_seconds=8.0,
    )
    report = collect_frame_directory_window(
        contract(),
        short_policy,
        {"status": "missing", "errors": ["not_configured"]},
        source,
        10.0,
        tmp_path / "scheduled",
    )
    assert report["window_status"] == "pass"
    assert report["sample_count"] == 5
    assert [row["source_index"] for row in report["sampled_frames"]] == [
        0,
        15,
        30,
        45,
        60,
    ]
    assert report["source"]["parent_metadata_read"] is False
    assert report["source"]["ground_truth_read"] is False
    assert report["geometry_handoff_status"] == "blocked"


def test_input_record_is_not_mutated_except_integer_normalization() -> None:
    item = record(7, 1.5, rgb=True)
    original_rgb = item.rgb.copy()
    buffer = UniformCameraWindowBuffer(policy(), contract())
    buffer.add(item)
    assert np.array_equal(item.rgb, original_rgb)
