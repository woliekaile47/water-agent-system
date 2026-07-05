#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[setup] project root: $PROJECT_ROOT"
echo "[setup] creating Python virtual environment at .venv"
python3 -m venv --system-site-packages .venv

echo "[setup] activating .venv"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] installing Python requirements"
python -m pip install -r requirements.txt

echo "[setup] done"
echo "[setup] ROS2 Humble is provided by the system environment."
echo "[setup] For ROS2 rosbag stages, manually run:"
echo "        source /opt/ros/humble/setup.bash"
echo "[setup] This script does not start ROS nodes, LiDAR, camera, or rosbag replay."
