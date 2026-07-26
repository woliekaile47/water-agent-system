#!/usr/bin/env bash
# Run one frozen simulated-sensor Agent scenario entirely inside WSL.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_simulation_agent_e2e_wsl.sh [options]

Options:
  --run-id ID              Safe unique run identifier.
  --config PATH            One-click scenario configuration.
  --project-root PATH      Repository root (defaults to the script parent).
  --sam2-venv PATH         SAM2 GPU virtual environment.
  --checkpoint PATH        SAM2.1 Hiera Tiny checkpoint.
  --resume-prepared        Reuse an already frozen prepare stage.
  -h, --help               Show this help.

This runner uses simulation inputs only. It does not start a real device,
send a real warning, call an external API, or read Ground Truth for prediction.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_ID="simulation_e2e_$(date -u +%Y%m%d_%H%M%S)"
CONFIG_PATH="configs/phase2d_c13_one_click_20cm.yaml"
SAM2_VENV="/home/wlkl/venvs/sam2-gpu"
CHECKPOINT="/home/wlkl/ai_models/sam2/checkpoints/sam2.1_hiera_tiny.pt"
RESUME_PREPARED=false

while (($#)); do
  case "$1" in
    --run-id)
      RUN_ID="${2:?missing value for --run-id}"
      shift 2
      ;;
    --config)
      CONFIG_PATH="${2:?missing value for --config}"
      shift 2
      ;;
    --project-root)
      PROJECT_ROOT="${2:?missing value for --project-root}"
      shift 2
      ;;
    --sam2-venv)
      SAM2_VENV="${2:?missing value for --sam2-venv}"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="${2:?missing value for --checkpoint}"
      shift 2
      ;;
    --resume-prepared)
      RESUME_PREPARED=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]]; then
  echo "run-id may contain only letters, digits, underscore and hyphen" >&2
  exit 2
fi
if [[ ! "$CONFIG_PATH" =~ ^configs/[A-Za-z0-9_.-]+\.yaml$ ]]; then
  echo "config must be a YAML file directly below configs/" >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd)"
CONFIG_FILE="$PROJECT_ROOT/$CONFIG_PATH"
SAM2_PYTHON="$SAM2_VENV/bin/python"
RUN_DIR="$PROJECT_ROOT/outputs/phase2d_c13_one_click_runs/$RUN_ID"
PAYLOAD_DIR="$RUN_DIR/exchange_to_wsl/payload"
RESULT_DIR="$RUN_DIR/result"
RESULT_ARCHIVE="$RUN_DIR/local_sam2_result.tar.gz"

test -f "$CONFIG_FILE"
test -x "$SAM2_PYTHON"
test -f "$CHECKPOINT"
set +u
source /opt/ros/humble/setup.bash
set -u

cd "$PROJECT_ROOT"
if [[ "$RESUME_PREPARED" == false ]]; then
  echo "[1/4] Preparing dry-LiDAR Ground DEM and automatic temporal prompt..."
  python3 scripts/run_simulation_agent_e2e_vm.py \
    --config "$CONFIG_PATH" \
    --project-root "$PROJECT_ROOT" \
    prepare --run-id "$RUN_ID"
else
  echo "[1/4] Reusing the frozen prepare stage..."
  test -s "$RUN_DIR/prepare_summary.json"
  test -s "$PAYLOAD_DIR/exchange_manifest.json"
fi

if [[ -e "$RESULT_DIR" || -e "$RESULT_ARCHIVE" ]]; then
  echo "Refusing to overwrite an existing local SAM2 result for $RUN_ID" >&2
  exit 1
fi

WINDOW_START="$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["window_start"])' \
    "$PAYLOAD_DIR/exchange_manifest.json"
)"
WINDOW_END="$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["window_end"])' \
    "$PAYLOAD_DIR/exchange_manifest.json"
)"
ANCHOR_FRAME="$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["anchor_frame_index"])' \
    "$PAYLOAD_DIR/exchange_manifest.json"
)"

echo "[2/4] Running frozen SAM2 video propagation on the WSL GPU..."
"$SAM2_PYTHON" "$PAYLOAD_DIR/run_sam2_video_propagation.py" \
  --frames-dir "$PAYLOAD_DIR/frames" \
  --prompt-config "$PAYLOAD_DIR/automatic_prompt.json" \
  --window-start "$WINDOW_START" \
  --window-end "$WINDOW_END" \
  --anchor-frame-index "$ANCHOR_FRAME" \
  --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$RESULT_DIR" \
  --device cuda

echo "[3/4] Freezing the SAM2 result and running geometry, quality gate and Agent..."
tar -czf "$RESULT_ARCHIVE" -C "$RUN_DIR" result
python3 scripts/run_simulation_agent_e2e_vm.py \
  --config "$CONFIG_PATH" \
  --project-root "$PROJECT_ROOT" \
  finalize --run-id "$RUN_ID" --sam2-archive "$RESULT_ARCHIVE"

echo "[4/4] Completed: $RUN_DIR/completion_summary.json"
echo "Safety: simulation-only, non-authoritative, no real warning or device action."
