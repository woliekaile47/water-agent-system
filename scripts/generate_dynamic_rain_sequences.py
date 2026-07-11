#!/usr/bin/env python3
"""Generate deterministic Phase 2C-1 dynamic rain visual sequences."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.synthetic_rain_visual_generator import (
    extract_stable_rgb_frame_from_rosbag,
    generate_sequence,
)


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    return document["dynamic_rain_visual_simulation"]


def choose_bag(project_root: Path, pattern: str) -> Path:
    candidates = sorted(path for path in project_root.glob(pattern) if path.is_dir() and (path / "metadata.yaml").is_file())
    if not candidates:
        raise FileNotFoundError(f"No rosbag directory matches: {pattern}")
    return candidates[0]


def load_water_mask(project_root: Path, case_id: str, expected_shape: tuple[int, int]) -> np.ndarray:
    path = project_root / "data" / "simulation" / case_id / "ground_truth" / "camera_water_mask_gt.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127
    if mask.shape != expected_shape:
        raise ValueError(f"Camera mask shape {mask.shape} does not match base RGB {expected_shape}")
    return mask


def write_dataset_summary(root: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "case_id", "rain_level", "random_seed", "frame_count", "fps", "duration_s",
        "dry_event_count", "water_event_count", "total_event_count", "quality_status",
        "sequence_sha256", "output_dir",
    ]
    rows = []
    for result in results:
        manifest = result["manifest"]
        rows.append({
            "case_id": manifest["case_id"],
            "rain_level": manifest["rain_level"],
            "random_seed": manifest["random_seed"],
            "frame_count": manifest["frame_count"],
            "fps": manifest["fps"],
            "duration_s": manifest["duration_s"],
            "dry_event_count": manifest["dry_event_count"],
            "water_event_count": manifest["water_event_count"],
            "total_event_count": manifest["dry_event_count"] + manifest["water_event_count"],
            "quality_status": manifest["generation_quality_status"],
            "sequence_sha256": manifest["sequence_sha256"],
            "output_dir": result["output_dir"],
        })
    with (root / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Phase 2C-1 dynamic rain visual dataset summary",
        "",
        "Synthetic visual abstraction only; not validated fluid dynamics.",
        "",
        "| case | rain | seed | frames | dry events | water events | total | quality |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['rain_level']} | {row['random_seed']} | {row['frame_count']} | "
            f"{row['dry_event_count']} | {row['water_event_count']} | {row['total_event_count']} | {row['quality_status']} |"
        )
    (root / "dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--config", default="configs/dynamic_rain_visual_simulation.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--case")
    parser.add_argument("--rain-level", choices=("light", "moderate", "heavy"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_config(config_path)
    bag = choose_bag(project_root, str(config["base_image"]["bag_glob"]))
    base_rgb, base_source = extract_stable_rgb_frame_from_rosbag(
        bag,
        str(config["base_image"]["topic"]),
        int(config["base_image"]["skip_camera_frames"]),
        int(config["base_image"]["median_frame_count"]),
    )
    base_output = project_root / "outputs" / "dynamic_visual_simulation"
    base_output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base_rgb, mode="RGB").save(base_output / "base_rgb_from_dry_rosbag.png")
    if args.all:
        selections = [
            (case_id, rain_level, int(config["rain_levels"][rain_level]["random_seed"]))
            for case_id in config["cases"]
            for rain_level in ("light", "moderate", "heavy")
        ]
    else:
        if args.rain_level is None:
            parser.error("--case requires --rain-level")
        seed = int(args.seed) if args.seed is not None else int(config["rain_levels"][args.rain_level]["random_seed"])
        selections = [(str(args.case), str(args.rain_level), seed)]
    output_root = project_root / str(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for case_id, rain_level, seed in selections:
        mask = load_water_mask(project_root, case_id, base_rgb.shape[:2])
        output = output_root / case_id / rain_level / f"seed_{seed}"
        result = generate_sequence(
            base_rgb,
            mask,
            config,
            case_id,
            rain_level,
            seed,
            output,
            base_source,
            project_root,
            create_preview=not args.no_preview,
        )
        results.append(result)
        manifest = result["manifest"]
        print(
            f"[phase2c1] {case_id}/{rain_level}/seed_{seed}: "
            f"frames={manifest['frame_count']} dry={manifest['dry_event_count']} "
            f"water={manifest['water_event_count']} quality={manifest['generation_quality_status']}"
        )
    if args.all:
        write_dataset_summary(output_root, results)


if __name__ == "__main__":
    main()
