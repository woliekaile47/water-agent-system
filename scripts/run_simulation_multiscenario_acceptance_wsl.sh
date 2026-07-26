#!/usr/bin/env bash
# Run the frozen 5/10/20/40 cm simulation acceptance matrix inside WSL.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_simulation_multiscenario_acceptance_wsl.sh [options]

Options:
  --matrix-id ID           Safe unique matrix identifier.
  --project-root PATH      Repository root (defaults to the script parent).
  --sam2-venv PATH         SAM2 GPU virtual environment.
  --checkpoint PATH        SAM2.1 Hiera Tiny checkpoint.
  -h, --help               Show this help.

The four scenarios remain simulation-only and Ground-Truth-free on the
prediction side. Expected quality-gate blocks are counted as safe outcomes.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MATRIX_ID="simulation_matrix_$(date -u +%Y%m%d_%H%M%S)"
SAM2_VENV="/home/wlkl/venvs/sam2-gpu"
CHECKPOINT="/home/wlkl/ai_models/sam2/checkpoints/sam2.1_hiera_tiny.pt"

while (($#)); do
  case "$1" in
    --matrix-id)
      MATRIX_ID="${2:?missing value for --matrix-id}"
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

if [[ ! "$MATRIX_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,60}$ ]]; then
  echo "matrix-id may contain only letters, digits, underscore and hyphen" >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd)"
RUNNER="$PROJECT_ROOT/scripts/run_simulation_agent_e2e_wsl.sh"
test -x "$RUNNER"

LABELS=(5cm 10cm 20cm 40cm)
CONFIGS=(
  configs/phase2d_c15_one_click_5cm.yaml
  configs/phase2d_c15_one_click_10cm.yaml
  configs/phase2d_c13_one_click_20cm.yaml
  configs/phase2d_c15_one_click_40cm.yaml
)

for index in "${!LABELS[@]}"; do
  label="${LABELS[$index]}"
  config="${CONFIGS[$index]}"
  echo "=== WSL-local acceptance: $label ==="
  "$RUNNER" \
    --run-id "${MATRIX_ID}_${label}" \
    --config "$config" \
    --project-root "$PROJECT_ROOT" \
    --sam2-venv "$SAM2_VENV" \
    --checkpoint "$CHECKPOINT"
done

set +u
source /opt/ros/humble/setup.bash
set -u
cd "$PROJECT_ROOT"
python3 scripts/summarize_phase2d_c15_matrix.py \
  --project-root "$PROJECT_ROOT" \
  --matrix-id "$MATRIX_ID"
echo "Matrix report: $PROJECT_ROOT/outputs/phase2d_c15_multiscenario_acceptance/$MATRIX_ID"
