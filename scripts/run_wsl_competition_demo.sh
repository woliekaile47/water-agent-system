#!/usr/bin/env bash
# Build or refresh the WSL-only simulation demo and start its Dashboard.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_wsl_competition_demo.sh [options]

Options:
  --matrix-id ID      Display a specific completed safe matrix.
  --refresh           Run a new 5/10/20/40 cm WSL matrix before display.
  --port PORT         Streamlit port (default: 8501).
  --address ADDRESS   Listen address (default: 0.0.0.0).
  --headless BOOL     Streamlit headless mode (default: true).
  -h, --help          Show this help.

Without --refresh, the newest completed safe WSL matrix is displayed quickly.
With --refresh, the full simulation Agent pipeline runs before the Dashboard.
No VMware, SSH transfer, real device, Ground Truth prediction, external API,
or real warning action is used.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MATRIX_ID=""
REFRESH=false
PORT=8501
ADDRESS="0.0.0.0"
HEADLESS=true

while (($#)); do
  case "$1" in
    --matrix-id)
      MATRIX_ID="${2:?missing value for --matrix-id}"
      shift 2
      ;;
    --refresh)
      REFRESH=true
      shift
      ;;
    --port)
      PORT="${2:?missing value for --port}"
      shift 2
      ;;
    --address)
      ADDRESS="${2:?missing value for --address}"
      shift 2
      ;;
    --headless)
      HEADLESS="${2:?missing value for --headless}"
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

if [[ -n "$MATRIX_ID" && ! "$MATRIX_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]]; then
  echo "matrix-id contains unsafe characters" >&2
  exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]{2,5}$ ]] || ((PORT < 1024 || PORT > 65535)); then
  echo "port must be between 1024 and 65535" >&2
  exit 2
fi
if [[ "$HEADLESS" != true && "$HEADLESS" != false ]]; then
  echo "headless must be true or false" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
if [[ "$REFRESH" == true ]]; then
  if [[ -z "$MATRIX_ID" ]]; then
    MATRIX_ID="wsl_competition_$(date -u +%Y%m%d_%H%M%S)"
  fi
  echo "[1/3] Running the complete WSL-local 5/10/20/40 cm matrix..."
  scripts/run_simulation_multiscenario_acceptance_wsl.sh --matrix-id "$MATRIX_ID"
else
  echo "[1/3] Reusing a completed safe WSL-local matrix..."
fi

echo "[2/3] Building the display-only Dashboard snapshot..."
builder=(python3 scripts/build_wsl_competition_demo_snapshot.py)
if [[ -n "$MATRIX_ID" ]]; then
  builder+=(--matrix-id "$MATRIX_ID")
fi
"${builder[@]}"

echo "[3/3] Starting the WSL-only competition Dashboard..."
echo "Open http://localhost:$PORT/ in the Windows browser."
echo "Dashboard display is non-authoritative and cannot send a real warning."
exec python3 -m streamlit run dashboard/competition_demo.py \
  --server.address "$ADDRESS" \
  --server.port "$PORT" \
  --server.headless "$HEADLESS" \
  --browser.gatherUsageStats false
