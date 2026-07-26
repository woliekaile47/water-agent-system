#!/usr/bin/env python3
"""Build the Dashboard snapshot from a completed WSL-local acceptance matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.wsl_competition_demo_snapshot import (  # noqa: E402
    build_wsl_competition_demo_snapshot,
    find_latest_completed_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-id",
        help="Completed safe matrix ID; defaults to the newest passing local matrix.",
    )
    parser.add_argument(
        "--matrix-config",
        default="configs/phase2d_c15_multiscenario_acceptance.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2d_wsl_competition_demo_snapshot",
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    matrix_id = args.matrix_id or find_latest_completed_matrix(root)
    snapshot = build_wsl_competition_demo_snapshot(
        root, matrix_id, args.matrix_config
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "competition_demo_snapshot.json"
    output_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"matrix_id: {matrix_id}")
    print(f"competition demo snapshot: {output_path}")
    print(f"cases: {snapshot['case_count']}")
    print("runtime: WSL local simulation Agent end-to-end")
    print("ground_truth_used: false")
    print("authoritative: false")
    print("eligible_for_real_warning: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
