#!/usr/bin/env python3
"""Generate a frozen SAM 2 prompt from existing GT-free temporal artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.generate_temporal_sam2_prompt import generate_temporal_sam2_prompt, sha256_file


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    return document["temporal_sam2_prompt"]


def _load_mask(prediction_dir: Path, names: tuple[str, ...]) -> tuple[np.ndarray, Path]:
    for name in names:
        path = prediction_dir / name
        if path.is_file():
            return np.asarray(Image.open(path).convert("L")) > 0, path
    raise FileNotFoundError(f"none of the mask artifacts exist: {', '.join(names)}")


def _load_classifications(prediction_dir: Path) -> tuple[list[dict[str, Any]], Path | None]:
    path = prediction_dir / "event_classifications.json"
    if path.is_file():
        return list(_load_json(path).get("classifications", [])), path
    path = prediction_dir / "temporal_diagnostics.json"
    if path.is_file():
        return list(_load_json(path).get("classifications", [])), path
    return [], None


def _load_gate(prediction_dir: Path) -> tuple[dict[str, Any], Path]:
    for name in ("visual_quality_gate.json", "quality_gate.json"):
        path = prediction_dir / name
        if path.is_file():
            return _load_json(path), path
    raise FileNotFoundError("visual_quality_gate.json or quality_gate.json is required")


def _save_preview(path: Path, image: Image.Image, prompt: dict[str, Any]) -> None:
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    box = prompt.get("box_xyxy")
    if box is not None:
        draw.rectangle(tuple(box), outline=(255, 215, 0), width=2)
    for index, (x, y) in enumerate(prompt["positive_points_xy"], 1):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(0, 255, 0), outline=(255, 255, 255))
        draw.text((x + 5, y - 7), f"P{index}", fill=(0, 255, 0))
    for index, (x, y) in enumerate(prompt["negative_points_xy"], 1):
        draw.line((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 0), width=2)
        draw.line((x - 4, y + 4, x + 4, y - 4), fill=(255, 0, 0), width=2)
        draw.text((x + 5, y - 7), f"N{index}", fill=(255, 0, 0))
    preview.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--frame-index", required=True, type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "temporal_sam2_prompt.yaml",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_dir = args.prediction_dir.resolve()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    probability = np.load(prediction_dir / "predicted_water_probability.npy")
    water, water_path = _load_mask(
        prediction_dir, ("predicted_camera_water_mask.png", "predicted_water_mask.png")
    )
    unknown, unknown_path = _load_mask(
        prediction_dir, ("predicted_camera_unknown_mask.png", "predicted_unknown_mask.png")
    )
    classifications, classifications_path = _load_classifications(prediction_dir)
    gate, gate_path = _load_gate(prediction_dir)
    if probability.shape != (image.height, image.width):
        raise ValueError("prediction artifacts do not match the reference image dimensions")
    prompt, diagnostics = generate_temporal_sam2_prompt(
        probability,
        water,
        unknown,
        classifications,
        gate,
        _load_config(args.config),
        image_path=str(image_path),
        image_sha256=sha256_file(image_path),
        frame_index=args.frame_index,
    )
    prompt["prediction_artifact_dir"] = str(prediction_dir)
    artifact_paths = [
        prediction_dir / "predicted_water_probability.npy",
        water_path,
        unknown_path,
        gate_path,
        args.config.resolve(),
    ]
    if classifications_path is not None:
        artifact_paths.append(classifications_path)
    prompt["prediction_artifact_sha256"] = {
        str(path.resolve()): sha256_file(path) for path in artifact_paths
    }
    _write_json(output_dir / "automatic_prompt.json", prompt)
    _write_json(output_dir / "automatic_prompt_diagnostics.json", diagnostics)
    _save_preview(output_dir / "automatic_prompt_preview.png", image, prompt)
    print(json.dumps({
        "status": prompt["prompt_quality_status"],
        "positive_points": len(prompt["positive_points_xy"]),
        "negative_points": len(prompt["negative_points_xy"]),
        "output_dir": str(output_dir),
        "ground_truth_used": False,
    }, ensure_ascii=False))
    return 0 if prompt["prompt_quality_status"] != "reject" else 2


if __name__ == "__main__":
    raise SystemExit(main())
