#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build one 60-second/41-frame Camera window without starting device nodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sensors.camera_window_scheduler import run_camera_window_scheduler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-config",
        default="configs/camera_window_scheduler.yaml",
        help="60-second Camera scheduling policy",
    )
    parser.add_argument(
        "--sensor-config",
        default="configs/unified_sensor_input.yaml",
        help="Shared rosbag/live topic and frame contract",
    )
    parser.add_argument(
        "--input-mode",
        required=True,
        choices=("rosbag", "live_topics", "frame_directory"),
        help="Read an existing bag or observe an already-published Camera topic",
    )
    parser.add_argument("--bag", help="ROS2 bag directory in rosbag mode")
    parser.add_argument(
        "--frames-dir",
        help=(
            "GT-isolated simulation frames/ directory in frame_directory mode"
        ),
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        help="Explicit simulation frame rate in frame_directory mode",
    )
    parser.add_argument(
        "--calibration",
        help="Independent real Camera/map calibration YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/camera_window_runtime/current",
        help="Generated frames and readiness report",
    )
    parser.add_argument(
        "--require-geometry",
        action="store_true",
        help="Return non-zero unless independent calibration permits geometry",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.input_mode == "rosbag" and not args.bag:
        raise SystemExit("--bag is required when --input-mode=rosbag")
    if args.input_mode == "frame_directory" and (
        not args.frames_dir or args.source_fps is None
    ):
        raise SystemExit(
            "--frames-dir and --source-fps are required "
            "when --input-mode=frame_directory"
        )
    try:
        report = run_camera_window_scheduler(
            window_config_path=args.window_config,
            sensor_config_path=args.sensor_config,
            input_mode=args.input_mode,
            output_dir=args.output_dir,
            bag_path=args.bag,
            calibration_path=args.calibration,
            frames_dir=args.frames_dir,
            source_fps=args.source_fps,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        failure = {
            "window_status": "reject",
            "pipeline_status": "blocked",
            "error_code": "camera_window_source_or_config_error",
            "message": str(exc),
            "recommended_action": (
                "检查输入路径、Camera topic、ROS2 环境、配置和图像编码后重试。"
            ),
            "formal_water_depth_generated": False,
            "ground_truth_used": False,
        }
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
        return 4
    problems = report["window_problems"] + report["geometry_problems"]
    summary = {
        "window_status": report["window_status"],
        "pipeline_status": report["pipeline_status"],
        "source_frame_count": report["source_frame_count"],
        "observed_span_seconds": report["observed_span_seconds"],
        "sample_count": report["sample_count"],
        "effective_sample_rate_hz": report["effective_sample_rate_hz"],
        "temporal_sam2_handoff_allowed": report[
            "temporal_sam2_handoff_allowed"
        ],
        "geometry_handoff_status": report["geometry_handoff_status"],
        "depth_inversion_allowed": report["depth_inversion_allowed"],
        "formal_water_depth_generated": report[
            "formal_water_depth_generated"
        ],
        "problems": problems,
        "readiness_path": report["readiness_path"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    for item in problems:
        print(
            f"[{item['code']}] {item['message']} "
            f"建议：{item['recommended_action']}",
            file=sys.stderr,
        )
    if report["window_status"] != "pass":
        return 2
    if args.require_geometry and report["geometry_handoff_status"] != "ready":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
