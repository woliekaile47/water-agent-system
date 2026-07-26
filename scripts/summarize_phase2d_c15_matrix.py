#!/usr/bin/env python3
"""Write the Phase 2D-C-15 four-scenario acceptance report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.simulation_acceptance_matrix import (
    load_matrix_config,
    render_markdown,
    summarize_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--config", default="configs/phase2d_c15_multiscenario_acceptance.yaml"
    )
    parser.add_argument("--matrix-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_matrix_config(config_path)
    summary = summarize_matrix(root, config_path, args.matrix_id)
    output_dir = root / config["output_root"] / args.matrix_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "acceptance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "acceptance_report.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
