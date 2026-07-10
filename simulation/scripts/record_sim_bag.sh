#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <case_id> [project_root]" >&2
  exit 2
fi

CASE_ID="$1"
PROJECT_ROOT="${2:-$(pwd)}"
OUTPUT_DIR="${PROJECT_ROOT}/data/simulation/${CASE_ID}/rosbag"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BAG_PATH="${OUTPUT_DIR}/${CASE_ID}__${STAMP}"

case "${CASE_ID}" in
  sim_dry_baseline_001)
    TOPICS=(
      /clock /tf /tf_static
      /sim/lidar/points
      /sim/camera/image_raw /sim/camera/camera_info
      /sim/ground_truth/water_level
      /sim/ground_truth/water_mask
      /sim/ground_truth/depth_map
    )
    ;;
  sim_water_5cm_001|sim_water_10cm_001|sim_water_20cm_001|sim_water_40cm_001)
    TOPICS=(
      /clock /tf /tf_static
      /sim/camera/image_raw /sim/camera/camera_info
      /sim/ground_truth/water_level
      /sim/ground_truth/water_mask
      /sim/ground_truth/depth_map
    )
    ;;
  *)
    echo "Unknown Phase 1 case: ${CASE_ID}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}"
echo "Recording simulation-only topics to ${BAG_PATH}"
echo "No real /cx or /hik_camera topics are recorded."
ros2 bag record -o "${BAG_PATH}" "${TOPICS[@]}"
