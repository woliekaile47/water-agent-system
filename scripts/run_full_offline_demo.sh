#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[demo] project root: $PROJECT_ROOT"
echo "[demo] running offline Agent pipeline"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "[demo] using .venv"
fi

python3 src/agent/pipeline_agent.py --config configs/agent_config.yaml

echo "[demo] showing SQLite audit database summary"
python3 src/database/show_audit_db.py --db data/db/water_agent_audit.db

echo "[demo] key output paths"
for path in \
  "outputs/json/agent_run_summary.json" \
  "outputs/json/final_forecast_result.json" \
  "outputs/json/warning_decision_result.json" \
  "outputs/reports/warning_report.md" \
  "outputs/figures/warning_summary.png"
do
  if [ -e "$path" ]; then
    echo "  [ok] $path"
  else
    echo "  [missing] $path"
  fi
done

echo "[demo] complete"
echo "[demo] This script only runs the offline pipeline."
echo "[demo] It does not start LiDAR, camera, ROS nodes, or rosbag replay."
