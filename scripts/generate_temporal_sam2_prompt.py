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
from src.vision.temporal_sam2_prompt_pipeline import run_temporal_sam2_prompt_from_frames


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _load_config_document(path: Path, key: str) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    return document[key]


def _load_config(path: Path) -> dict[str, Any]:
    return _load_config_document(path, "temporal_sam2_prompt")


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


def _save_binary(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def _save_prediction_artifacts(path: Path, result: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    prediction = result["prediction"]
    evidence = prediction["evidence"]
    np.save(path / "predicted_water_probability.npy", evidence["predicted_water_probability"])
    np.save(path / "evidence_count_map.npy", evidence["evidence_count_map"])
    _save_binary(path / "predicted_camera_water_mask.png", evidence["predicted_water_mask"])
    _save_binary(path / "predicted_camera_unknown_mask.png", evidence["predicted_unknown_mask"])
    _write_json(path / "event_classifications.json", {
        "data_role": "prediction",
        "classifications": prediction["classifications"],
        "ground_truth_used": False,
    })
    _write_json(path / "temporal_diagnostics.json", {
        "loader": prediction["loader"],
        "preprocessing": prediction["preprocessing_diagnostics"],
        "candidates": prediction["candidate_diagnostics"],
        "evidence": prediction["evidence_diagnostics"],
        "water_mask_time_stability": prediction["water_mask_time_stability"],
        "feature_score_separation": prediction["feature_score_separation"],
        "order_sensitivity": result["order_sensitivity"],
        "ground_truth_used": False,
    })
    _write_json(path / "visual_quality_gate.json", result["temporal_quality_gate"])
    _write_json(path / "prediction_manifest.json", {
        "data_role": "prediction",
        "source": "temporal_water_evidence_for_sam2_prompt",
        "detector_input_role": "continuous_rgb_frames_only",
        "ground_truth_or_metadata_read_during_prediction": False,
        "quality_gate_status": result["temporal_quality_gate"]["status"],
        "result_semantics": "sparse_temporal_evidence_for_prompt_generation",
        "eligible_for_downstream": False,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prediction-dir", type=Path)
    source.add_argument("--frames-dir", type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--frame-index", required=True, type=int)
    parser.add_argument("--expected-image-sha256")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "temporal_sam2_prompt.yaml",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_sha256 = sha256_file(image_path)
    if args.expected_image_sha256 and image_sha256 != args.expected_image_sha256.lower():
        raise ValueError("reference image SHA-256 does not match the frozen manifest")
    prompt_config = _load_config(args.config)

    generated_result: dict[str, Any] | None = None
    if args.frames_dir is not None:
        generated_result = run_temporal_sam2_prompt_from_frames(
            args.frames_dir,
            image_path,
            args.frame_index,
            _load_config_document(args.detector_config, "temporal_water_mask_detector"),
            _load_config_document(args.temporal_gate_config, "temporal_water_quality_gate"),
            prompt_config,
            expected_image_sha256=args.expected_image_sha256,
        )
        prediction_dir = output_dir / "temporal_prediction"
        _save_prediction_artifacts(prediction_dir, generated_result)
    else:
        prediction_dir = args.prediction_dir.resolve()
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
    if generated_result is None:
        prompt, diagnostics = generate_temporal_sam2_prompt(
            probability,
            water,
            unknown,
            classifications,
            gate,
            prompt_config,
            image_path=str(image_path),
            image_sha256=image_sha256,
            frame_index=args.frame_index,
        )
    else:
        prompt = generated_result["prompt"]
        diagnostics = generated_result["prompt_diagnostics"]
    diagnostics["reference_image_sha256_verified"] = args.expected_image_sha256 is not None
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
