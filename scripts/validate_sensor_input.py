#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate one rosbag or live ROS2 source against the shared sensor contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sensors.unified_sensor_input import validate_sensor_input


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate sensor ingest only. This command never runs water-mask, "
            "DEM fusion, depth inversion, warning, or Ground Truth evaluation."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/unified_sensor_input.yaml",
        help="Shared sensor input contract YAML",
    )
    parser.add_argument(
        "--input-mode",
        choices=("rosbag", "live_topics"),
        help="Override config transport without changing downstream contract",
    )
    parser.add_argument(
        "--operating-phase",
        choices=("baseline_scan", "rain_monitoring", "combined_validation"),
        help=(
            "Two-stage sensor profile: baseline_scan requires LiDAR, "
            "rain_monitoring requires Camera, combined_validation requires both"
        ),
    )
    parser.add_argument("--bag", help="ROS2 bag directory for rosbag mode")
    parser.add_argument(
        "--calibration",
        help="Independent real calibration YAML; never use simulation GT calibration",
    )
    parser.add_argument(
        "--live-timeout-seconds",
        type=float,
        help="How long live mode waits for one camera and one LiDAR message",
    )
    parser.add_argument(
        "--output",
        default="outputs/sensor_input_validation/readiness.json",
        help="Generated readiness JSON",
    )
    parser.add_argument(
        "--require-geometry",
        action="store_true",
        help="Return non-zero unless independent calibration permits geometry",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_sensor_input(
        args.config,
        input_mode=args.input_mode,
        operating_phase=args.operating_phase,
        bag_path=args.bag,
        calibration_path=args.calibration,
        live_timeout_seconds=args.live_timeout_seconds,
        output_path=args.output,
    )
    summary = {
        "source_mode": report["source_mode"],
        "operating_phase": report["operating_phase"],
        "data_ingest_status": report["data_ingest_status"],
        "geometry_readiness": report["geometry_readiness"],
        "depth_inversion_allowed": report["depth_inversion_allowed"],
        "next_allowed_stage": report["next_allowed_stage"],
        "blocked_reasons": report["blocked_reasons"],
        "ground_truth_used": report["ground_truth_used"],
        "output": str(Path(args.output).expanduser().resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if report["data_ingest_status"] != "pass":
        return 2
    if args.require_geometry and report["geometry_readiness"] != "ready":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
