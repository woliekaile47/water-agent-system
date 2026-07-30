#!/usr/bin/env python3
"""Independently evaluate one frozen dense-burst temporal mask against Camera GT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_temporal_sam2_mask_gt import (  # noqa: E402
    evaluate_camera_mask,
    load_camera_mask_ground_truth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--prediction-mask", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--rain-level", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-frame-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_comparison(predicted: np.ndarray, truth: np.ndarray, path: Path) -> None:
    image = np.zeros((*predicted.shape, 3), dtype=np.uint8)
    image[:] = (35, 35, 35)
    image[predicted & truth] = (40, 190, 80)
    image[predicted & ~truth] = (225, 70, 55)
    image[~predicted & truth] = (50, 120, 235)
    Image.fromarray(image, mode="RGB").save(path)


def verify_frozen_prediction(args: argparse.Namespace) -> dict[str, Any]:
    selection_hash = sha256_file(args.selection_summary)
    if selection_hash != args.expected_selection_sha256.lower():
        raise ValueError("frozen dense-burst selection hash mismatch")
    selection = read_json(args.selection_summary)
    prompt = read_json(args.prompt)
    if selection.get("ground_truth_used") is not False:
        raise ValueError("selection artifact is not prediction-only")
    if prompt.get("ground_truth_used") is not False:
        raise ValueError("prompt artifact is not prediction-only")
    if selection.get("selection", {}).get("selection_status") != "reject":
        raise ValueError("this evaluator expects the frozen fail-closed selection result")
    predicted_raw = np.asarray(Image.open(args.prediction_mask).convert("L"))
    if predicted_raw.shape != (360, 640):
        raise ValueError("prediction mask shape is not 360x640")
    return {
        "selection_summary_sha256": selection_hash,
        "prediction_mask_sha256": sha256_file(args.prediction_mask),
        "prompt_sha256": sha256_file(args.prompt),
        "predicted_mask": predicted_raw > 127,
        "verified_before_ground_truth_read": True,
    }


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")

    # Protocol boundary: verify every frozen prediction artifact before the first GT read.
    frozen = verify_frozen_prediction(args)
    gt = load_camera_mask_ground_truth(
        args.project_root,
        args.case_id,
        args.rain_level,
        args.seed,
        args.source_frame_index,
    )
    truth = np.asarray(gt["camera_mask"], dtype=bool)
    result = evaluate_camera_mask(frozen.pop("predicted_mask"), truth)
    result.update(
        {
            "evaluation_role": "independent_dense_burst_temporal_mask_gt_evaluation",
            "frozen_prediction_verification": frozen,
            "ground_truth_validation": gt["validation"],
            "selection_status": "reject",
            "sam2_run_count": 0,
            "prediction_recomputed_after_gt": False,
            "ground_truth_used_for_prediction": False,
            "authoritative": False,
            "eligible_for_downstream": False,
        }
    )
    output.mkdir(parents=True)
    write_json(output / "evaluation_summary.json", result)
    predicted = np.asarray(Image.open(args.prediction_mask).convert("L")) > 127
    save_comparison(predicted, truth, output / "temporal_mask_vs_gt.png")
    (output / "run_log.txt").write_text(
        "Frozen prediction verified before GT read. SAM2 was not run.\n",
        encoding="utf-8",
    )
    camera = result["camera_mask_metrics"]
    print(
        json.dumps(
            {
                "iou": camera["iou"],
                "precision": camera["precision"],
                "recall": camera["recall"],
                "predicted_pixels": camera["predicted_pixels"],
                "gt_pixels": camera["gt_pixels"],
                "output_dir": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
