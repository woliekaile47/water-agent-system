#!/usr/bin/env python3
"""Run the VM half of the one-click simulated-sensor Agent workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.simulation_agent_e2e_orchestrator import (  # noqa: E402
    finalize_run,
    prepare_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/phase2d_c13_one_click_20cm.yaml",
        help="One-click orchestration configuration",
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build DEM/prompt and WSL exchange archive")
    prepare.add_argument("--run-id", required=True)

    finalize = subparsers.add_parser(
        "finalize", help="Import frozen SAM2 output and run standard S4-S8/Agent"
    )
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--sam2-archive", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        result = prepare_run(args.project_root, args.config, args.run_id)
    else:
        result = finalize_run(
            args.project_root,
            args.config,
            args.run_id,
            args.sam2_archive,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
