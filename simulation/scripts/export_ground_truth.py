#!/usr/bin/env python3
"""Source-tree wrapper for deterministic Ground Truth export."""

from pathlib import Path
import sys

SIMULATION_ROOT = Path(__file__).resolve().parents[1]
if str(SIMULATION_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATION_ROOT))

from waterlogging_simulation.cli import export_ground_truth_main


if __name__ == "__main__":
    export_ground_truth_main()
