#!/usr/bin/env python3
"""Run one prediction-only Phase 2D-A synthetic RGB-to-depth sequence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.synthetic_visual_to_depth import run_synthetic_visual_to_depth_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    sequence = Path(args.sequence_dir).expanduser()
    if not sequence.is_absolute():
        sequence = root / sequence
    sequence = sequence.resolve()
    frames = sequence / "frames"
    if not sequence.is_dir() or not frames.is_dir():
        raise FileNotFoundError(f"Sequence or frames directory does not exist: {sequence}")
    if not sequence.name.startswith("seed_") or sequence.parent.name not in ("light", "moderate", "heavy"):
        raise ValueError("sequence-dir must end in <case_id>/<rain_level>/seed_<seed>")
    case_id, rain_level, seed_name = sequence.parents[1].name, sequence.parent.name, sequence.name
    config_path = root / "configs/synthetic_visual_to_depth_integration.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["synthetic_visual_to_depth_integration"]
    if args.output_dir:
        output = Path(args.output_dir).expanduser()
        if not output.is_absolute():
            output = root / output
    else:
        output = root / config["output_root"] / case_id / rain_level / seed_name
    result = run_synthetic_visual_to_depth_prediction(root, frames, output, config)
    geometry = result["geometry_result"]
    if geometry is None:
        failure = result["geometry_failure"]
        print(
            f"[phase2d-a] visual={result['visual_gate']['status']} geometry=unavailable "
            f"integration={result['integration_gate']['status']} "
            f"failure={failure['failure_stage']}:{failure['failure_type']} "
            "eligible_for_downstream=false"
        )
    else:
        print(
            f"[phase2d-a] visual={result['visual_gate']['status']} "
            f"geometry={result['geometry_gate']['status']} integration={result['integration_gate']['status']} "
            f"level_m={geometry['predicted_water_level_m']:.6f} "
            f"area_m2={geometry['water_area_m2']:.6f} volume_m3={geometry['water_volume_m3']:.6f} "
            "eligible_for_downstream=false"
        )


if __name__ == "__main__":
    main()
