"""Command-line entry points for scenario and Ground Truth generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generator import generate_case


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--scenario", required=True, help="Scenario ID from scenarios.yaml")
    parser.add_argument("--project-root", default=Path.cwd(), help="water_agent_system project root")
    parser.add_argument("--config-dir", help="Simulation config directory")
    parser.add_argument("--world-template", help="Path to low_lying_road.sdf template")
    parser.add_argument("--output-root", help="Optional output root instead of data/simulation")
    return parser


def _run(args: argparse.Namespace) -> dict:
    project_root = Path(args.project_root).expanduser().resolve()
    config_dir = (
        Path(args.config_dir).expanduser().resolve()
        if args.config_dir
        else project_root / "simulation" / "config"
    )
    world_template = (
        Path(args.world_template).expanduser().resolve()
        if args.world_template
        else project_root / "simulation" / "worlds" / "low_lying_road.sdf"
    )
    manifest = generate_case(
        case_id=args.scenario,
        project_root=project_root,
        config_dir=config_dir,
        world_template=world_template,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def set_scenario_main() -> None:
    args = _parser("Generate a resolved Gazebo world and Phase 1 Ground Truth").parse_args()
    _run(args)


def export_ground_truth_main() -> None:
    args = _parser("Export deterministic Phase 1 Ground Truth").parse_args()
    _run(args)
