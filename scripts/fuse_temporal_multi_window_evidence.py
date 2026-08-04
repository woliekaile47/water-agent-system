#!/usr/bin/env python3
"""Fuse recurring water evidence from fixed RGB bursts into one SAM 2 prompt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.temporal_dense_burst_selection import load_dense_burst_policy  # noqa: E402
from src.vision.temporal_multi_window_evidence_fusion import (  # noqa: E402
    load_multi_window_fusion_policy,
    run_temporal_multi_window_fusion,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument(
        "--window-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "temporal_dense_burst_selection.yaml",
    )
    parser.add_argument(
        "--fusion-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "temporal_multi_window_evidence_fusion.yaml",
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


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    result, artifacts = run_temporal_multi_window_fusion(
        args.frames_dir,
        args.source_fps,
        load_dense_burst_policy(args.window_config),
        load_multi_window_fusion_policy(args.fusion_config),
        load_section(args.detector_config, "temporal_water_mask_detector"),
        load_section(args.temporal_gate_config, "temporal_water_quality_gate"),
        load_section(args.prompt_config, "temporal_sam2_prompt"),
        output,
    )
    artifact_paths = {
        "cross_window_raw_support_count": output / "cross_window_raw_support_count.npy",
        "cross_window_support_count": output / "cross_window_support_count.npy",
        "cross_window_support_fraction": output / "cross_window_support_fraction.npy",
        "fused_evidence_probability": output / "fused_evidence_probability.npy",
        "fused_probability": output / "fused_probability.npy",
        "fused_water_mask": output / "fused_water_mask.npy",
        "fused_unknown_mask": output / "fused_unknown_mask.npy",
    }
    np.save(artifact_paths["cross_window_raw_support_count"], artifacts["raw_support_count"])
    np.save(artifact_paths["cross_window_support_count"], artifacts["support_count"])
    np.save(artifact_paths["cross_window_support_fraction"], artifacts["support_fraction"])
    np.save(artifact_paths["fused_evidence_probability"], artifacts["evidence_probability"])
    np.save(artifact_paths["fused_probability"], artifacts["fused_probability"])
    np.save(artifact_paths["fused_water_mask"], artifacts["fused_water_mask"])
    np.save(artifact_paths["fused_unknown_mask"], artifacts["fused_unknown_mask"])
    save_mask(output / "fused_water_mask.png", np.asarray(artifacts["fused_water_mask"], dtype=bool))
    save_mask(output / "fused_unknown_mask.png", np.asarray(artifacts["fused_unknown_mask"], dtype=bool))
    result["artifact_files"] = {
        key: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for key, path in artifact_paths.items()
    }
    write_json(output / "multi_window_fusion.json", result)
    if result["selected_prompt"] is not None:
        write_json(output / "automatic_prompt.json", result["selected_prompt"])
        write_json(output / "prompt_diagnostics.json", result["selected_prompt_diagnostics"])
    write_json(
        output / "fusion_manifest.json",
        {
            "schema_version": result["schema_version"],
            "algorithm_version": result["algorithm_version"],
            "source_frames_dir": result["source_frames_dir"],
            "source_frame_count": result["source_frame_count"],
            "selected_source_start_index": result["selected_source_start_index"],
            "multi_window_fusion_sha256": sha256_file(output / "multi_window_fusion.json"),
            "automatic_prompt_sha256": (
                sha256_file(output / "automatic_prompt.json")
                if (output / "automatic_prompt.json").is_file()
                else None
            ),
            "input_config_sha256": {
                "window": sha256_file(args.window_config),
                "fusion": sha256_file(args.fusion_config),
                "detector": sha256_file(args.detector_config),
                "temporal_gate": sha256_file(args.temporal_gate_config),
                "prompt": sha256_file(args.prompt_config),
            },
            "ground_truth_used": False,
            "manual_selection_used": False,
            "sam2_run_allowed": result["sam2_run_allowed"],
            "authoritative": False,
            "eligible_for_downstream": False,
        },
    )
    print(json.dumps({
        "fusion_status": result["fusion"]["fusion_status"],
        "eligible_windows": result["fusion"]["eligible_window_count"],
        "supporting_windows": result["fusion"]["selected_cluster_window_count"],
        "supporting_time_span_seconds": result["fusion"]["selected_cluster_time_span_seconds"],
        "positive_points": (
            len(result["selected_prompt"].get("positive_points_xy", []))
            if result["selected_prompt"] is not None
            else 0
        ),
        "sam2_run_allowed": result["sam2_run_allowed"],
        "output_dir": str(output),
    }, ensure_ascii=False))
    return 0 if result["sam2_run_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
