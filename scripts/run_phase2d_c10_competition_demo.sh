#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[competition-demo] validating frozen simulation-road inputs"
python3 scripts/build_phase2d_c10_competition_demo.py

echo "[competition-demo] starting the dedicated offline dashboard"
echo "[competition-demo] source policy: simulation road only"
echo "[competition-demo] no ROS, Gazebo, Camera, LiDAR, RTSP, Agent, or warning action is started"
exec streamlit run dashboard/competition_demo.py "$@"
