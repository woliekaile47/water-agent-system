#!/usr/bin/env python3
"""CLI for preparing the standard S4-to-S8 direct simulation runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.prepare_simulation_e2e_runtime import prepare_runtime  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/phase2d_c11_direct_e2e_20cm.yaml",
        help="Direct simulation run configuration",
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT, help="Repository code root")
    parser.add_argument("--runtime-root", help="Optional isolated runtime output root override")
    args = parser.parse_args()
    result = prepare_runtime(args.config, args.project_root, args.runtime_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
