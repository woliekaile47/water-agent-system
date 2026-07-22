#!/usr/bin/env python3
"""Build the frozen simulation-only competition demo snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.competition_demo_profile import build_competition_demo_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/phase2d_c10_competition_demo.yaml",
        help="Project-relative competition demo profile.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2d_c10_competition_demo_snapshot",
        help="Project-relative output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = build_competition_demo_snapshot(PROJECT_ROOT, args.config)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "competition_demo_snapshot.json"
    output_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"competition demo snapshot: {output_path}")
    print(f"cases: {snapshot['case_count']}")
    print("source policy: simulation road only; no dormitory/cardboard inputs")
    print("ground_truth_used: false")
    print("authoritative: false")
    print("eligible_for_downstream: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
