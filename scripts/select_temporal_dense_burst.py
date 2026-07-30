#!/usr/bin/env python3
"""Select one GT-free dense RGB burst for automatic temporal SAM 2 prompting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.temporal_dense_burst_selection import (  # noqa: E402
    load_dense_burst_policy,
    select_dense_burst,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "temporal_dense_burst_selection.yaml",
    )
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "temporal_water_mask_detector.yaml",
    )
    parser.add_argument(
        "--temporal-gate-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "temporal_water_quality_gate.yaml",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "temporal_sam2_prompt_corroborated.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_section(path: Path, key: str) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = document.get(key) if isinstance(document, dict) else None
    if not isinstance(section, dict):
        raise ValueError(f"{path} must contain a {key} mapping")
    return section


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    result = select_dense_burst(
        args.frames_dir,
        args.source_fps,
        load_dense_burst_policy(args.selection_config),
        load_section(args.detector_config, "temporal_water_mask_detector"),
        load_section(args.temporal_gate_config, "temporal_water_quality_gate"),
        load_section(args.prompt_config, "temporal_sam2_prompt"),
        args.output_dir,
    )
    output = args.output_dir.expanduser().resolve()
    write_json(output / "dense_burst_selection.json", result)
    if result["selected_prompt"] is not None:
        write_json(output / "automatic_prompt.json", result["selected_prompt"])
        write_json(
            output / "prompt_diagnostics.json",
            result["selected_prompt_diagnostics"],
        )
    print(
        json.dumps(
            {
                "selection_status": result["selection"]["selection_status"],
                "candidate_count": result["candidate_count"],
                "selected_candidate_index": result["selection"][
                    "selected_candidate_index"
                ],
                "sam2_run_allowed": result["sam2_run_allowed"],
                "output_dir": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["sam2_run_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
