#!/usr/bin/env python3
"""Rebuild only a SAM 2 prompt from immutable multi-window fusion artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.generate_temporal_sam2_prompt import (  # noqa: E402
    generate_temporal_sam2_prompt,
    sha256_file,
)


ARTIFACT_FILES = {
    "fused_probability": "fused_probability.npy",
    "fused_water_mask": "fused_water_mask.npy",
    "fused_unknown_mask": "fused_unknown_mask.npy",
    "cross_window_support_fraction": "cross_window_support_fraction.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "temporal_sam2_prompt_c22_packing_aware.yaml"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_prompt_config(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = document.get("temporal_sam2_prompt") if isinstance(document, dict) else None
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain temporal_sam2_prompt")
    return config


def git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def verify_recorded_hash(
    result: dict[str, Any],
    artifact_key: str,
    path: Path,
) -> str:
    actual = sha256_file(path)
    record = result.get("artifact_files", {}).get(artifact_key)
    if not isinstance(record, dict) or record.get("sha256") != actual:
        raise ValueError(f"frozen artifact hash mismatch: {artifact_key}")
    return actual


def main() -> int:
    args = parse_args()
    fusion_dir = args.fusion_dir.expanduser().resolve()
    prompt_config_path = args.prompt_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not fusion_dir.is_dir():
        raise FileNotFoundError(f"fusion directory not found: {fusion_dir}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite prompt-only output: {output_dir}")

    result_path = fusion_dir / "multi_window_fusion.json"
    manifest_path = fusion_dir / "fusion_manifest.json"
    result = read_json(result_path)
    source_manifest = read_json(manifest_path)
    if source_manifest.get("multi_window_fusion_sha256") != sha256_file(result_path):
        raise ValueError("frozen multi-window result hash mismatch")
    if result.get("ground_truth_used") is not False:
        raise ValueError("source result must explicitly declare ground_truth_used=false")
    if source_manifest.get("ground_truth_used") is not False:
        raise ValueError("source manifest must explicitly declare ground_truth_used=false")
    fusion = result.get("fusion")
    if not isinstance(fusion, dict) or fusion.get("fusion_status") != "pass":
        raise ValueError("frozen fusion must pass before prompt-only rebuild")

    paths = {key: fusion_dir / name for key, name in ARTIFACT_FILES.items()}
    artifact_hashes = {
        key: verify_recorded_hash(result, key, path)
        for key, path in paths.items()
    }
    probability = np.load(paths["fused_probability"], allow_pickle=False)
    water = np.load(paths["fused_water_mask"], allow_pickle=False)
    unknown = np.load(paths["fused_unknown_mask"], allow_pickle=False)
    support = np.load(paths["cross_window_support_fraction"], allow_pickle=False)
    shapes = {array.shape for array in (probability, water, unknown, support)}
    if len(shapes) != 1 or probability.ndim != 2:
        raise ValueError("frozen prompt arrays must be same-shape 2-D arrays")
    if not np.isfinite(probability).all() or not np.isfinite(support).all():
        raise ValueError("frozen prompt arrays contain non-finite values")

    anchor = int(result["anchor_frame_index"])
    image_path = fusion_dir / "frames" / f"frame_{anchor:06d}.png"
    image_sha256 = sha256_file(image_path)
    previous_prompt = result.get("selected_prompt")
    if isinstance(previous_prompt, dict) and previous_prompt.get("image_sha256") != image_sha256:
        raise ValueError("frozen anchor image hash mismatch")

    prompt, diagnostics = generate_temporal_sam2_prompt(
        probability,
        water,
        unknown,
        [],
        {
            "status": "pass",
            "reasons": [],
            "observable_region_result_valid": True,
            "ground_truth_used": False,
        },
        load_prompt_config(prompt_config_path),
        temporal_support_fraction=support,
        image_path=str(image_path),
        image_sha256=image_sha256,
        frame_index=anchor,
    )
    prompt.update({
        "prompt_source": "c22_frozen_multi_window_prompt_only_v1",
        "multi_window_fusion_status": fusion["fusion_status"],
        "supporting_window_count": fusion["selected_cluster_window_count"],
        "supporting_time_span_seconds": fusion["selected_cluster_time_span_seconds"],
        "representative_source_start_index": result["selected_source_start_index"],
        "ground_truth_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
    })
    diagnostics.update({
        "prompt_only_rebuild": True,
        "multi_window_fusion_status": fusion["fusion_status"],
        "ground_truth_used": False,
        "eligible_for_downstream": False,
    })
    sam2_run_allowed = prompt.get("prompt_quality_status") == "pass"

    output_dir.mkdir(parents=True)
    prompt_path = output_dir / "automatic_prompt.json"
    diagnostics_path = output_dir / "prompt_diagnostics.json"
    write_json(prompt_path, prompt)
    write_json(diagnostics_path, diagnostics)
    write_json(
        output_dir / "prompt_only_manifest.json",
        {
            "schema_version": "phase2d_c22_frozen_prompt_only_v1",
            "source_fusion_dir": str(fusion_dir),
            "source_fusion_manifest_sha256": sha256_file(manifest_path),
            "source_multi_window_fusion_sha256": sha256_file(result_path),
            "source_artifact_sha256": artifact_hashes,
            "source_anchor_frame": str(image_path),
            "source_anchor_frame_sha256": image_sha256,
            "prompt_config": str(prompt_config_path),
            "prompt_config_sha256": sha256_file(prompt_config_path),
            "prompt_implementation_sha256": sha256_file(
                PROJECT_ROOT / "src" / "vision" / "generate_temporal_sam2_prompt.py"
            ),
            "git_head": git_head(),
            "automatic_prompt_sha256": sha256_file(prompt_path),
            "prompt_diagnostics_sha256": sha256_file(diagnostics_path),
            "prompt_quality_status": prompt["prompt_quality_status"],
            "sam2_run_allowed": bool(sam2_run_allowed),
            "ground_truth_used": False,
            "manual_selection_used": False,
            "prompt_only": True,
            "authoritative": False,
            "eligible_for_downstream": False,
        },
    )
    print(json.dumps({
        "prompt_quality_status": prompt["prompt_quality_status"],
        "positive_points": len(prompt["positive_points_xy"]),
        "negative_points": len(prompt["negative_points_xy"]),
        "sam2_run_allowed": bool(sam2_run_allowed),
        "ground_truth_used": False,
        "output_dir": str(output_dir),
    }, ensure_ascii=False))
    return 0 if sam2_run_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
