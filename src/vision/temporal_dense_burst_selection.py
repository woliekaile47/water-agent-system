#!/usr/bin/env python3
"""GT-free deterministic dense-burst selection for temporal SAM 2 prompting."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

from src.vision.temporal_sam2_prompt_pipeline import run_temporal_sam2_prompt_from_frames


FORBIDDEN_CONFIG_TOKENS = (
    "ground_truth",
    "water_level_gt",
    "depth_map_gt",
    "dem_water_mask_gt",
    "camera_water_mask_gt",
    "nominal_depth",
    "area_gt",
    "volume_gt",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_keys(value: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{prefix}.{key}" if prefix else str(key)
            keys.append(current.lower())
            keys.extend(_flatten_keys(item, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            keys.extend(_flatten_keys(item, f"{prefix}[{index}]"))
    return keys


def load_dense_burst_policy(path: str | Path) -> dict[str, Any]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("dense-burst selection config root must be a mapping")
    forbidden = [
        key
        for key in _flatten_keys(document)
        if any(token in key for token in FORBIDDEN_CONFIG_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"prediction-side dense-burst config contains forbidden fields: {forbidden}")
    policy = document.get("temporal_dense_burst_selection")
    if not isinstance(policy, dict):
        raise ValueError("temporal_dense_burst_selection mapping is required")
    burst_count = int(policy["burst_frame_count"])
    anchor = int(policy["anchor_frame_index"])
    stride_seconds = float(policy["candidate_stride_seconds"])
    if burst_count < 20:
        raise ValueError("burst_frame_count must be at least 20")
    if not 0 <= anchor < burst_count:
        raise ValueError("anchor_frame_index must lie inside the burst")
    if stride_seconds <= 0.0:
        raise ValueError("candidate_stride_seconds must be positive")
    if policy.get("required_prompt_status") != "pass":
        raise ValueError("dense-burst selection must fail closed unless prompt status is pass")
    return policy


def enumerate_candidate_starts(
    source_frame_count: int,
    burst_frame_count: int,
    stride_frames: int,
) -> list[int]:
    if source_frame_count < burst_frame_count:
        return []
    if stride_frames < 1:
        raise ValueError("stride_frames must be positive")
    last = source_frame_count - burst_frame_count
    starts = list(range(0, last + 1, stride_frames))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _stage_frames(source_paths: list[Path], start: int, count: int, frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=False)
    for local_index, source in enumerate(source_paths[start : start + count]):
        shutil.copy2(source, frames_dir / f"frame_{local_index:06d}.png")


def _status_rank(status: str) -> int:
    return {"reject": 0, "diagnostic_only": 1, "pass": 2}.get(status, -1)


def summarize_candidate(
    result: dict[str, Any],
    *,
    candidate_index: int,
    source_start_index: int,
    source_end_index: int,
    source_fps: float,
) -> dict[str, Any]:
    prompt = result["prompt"]
    prompt_diagnostics = result["prompt_diagnostics"]
    gate = result["temporal_quality_gate"]
    prediction = result["prediction"]
    box_border_ratio = float(prompt_diagnostics.get("box_border_touch_ratio", 1.0))
    temporal_support = prompt_diagnostics.get("selected_positive_temporal_support", [])
    minimum_support = float(min(temporal_support)) if temporal_support else 0.0
    summary = {
        "candidate_index": int(candidate_index),
        "source_start_index": int(source_start_index),
        "source_end_index": int(source_end_index),
        "source_start_seconds": float(source_start_index / source_fps),
        "source_end_seconds": float(source_end_index / source_fps),
        "prompt_quality_status": str(prompt.get("prompt_quality_status", "reject")),
        "prompt_quality_reasons": list(prompt.get("prompt_quality_reasons", [])),
        "temporal_gate_status": str(gate.get("status", "reject")),
        "temporal_gate_reasons": list(gate.get("reasons", [])),
        "positive_point_count": len(prompt.get("positive_points_xy", [])),
        "negative_point_count": len(prompt.get("negative_points_xy", [])),
        "component_count": int(prompt_diagnostics.get("component_count", 0)),
        "selected_component_area_pixels": int(
            prompt_diagnostics.get("selected_component_area_pixels", 0)
        ),
        "box_touches_image_border": bool(
            prompt_diagnostics.get("box_touches_image_border", True)
        ),
        "box_border_touch_ratio": box_border_ratio,
        "minimum_selected_temporal_support": minimum_support,
        "order_sensitivity": float(result.get("order_sensitivity", 0.0)),
        "water_mask_time_stability": float(
            prediction.get("water_mask_time_stability", 0.0)
        ),
        "feature_score_separation": float(
            prediction.get("feature_score_separation", 0.0)
        ),
        "ground_truth_used": False,
        "manual_selection_used": False,
    }
    summary["selection_rank"] = [
        _status_rank(summary["prompt_quality_status"]),
        _status_rank(summary["temporal_gate_status"]),
        int(not summary["box_touches_image_border"]),
        int(summary["positive_point_count"] >= 3),
        int(summary["negative_point_count"] >= 6),
        summary["minimum_selected_temporal_support"],
        summary["water_mask_time_stability"],
        summary["feature_score_separation"],
        summary["order_sensitivity"],
        -summary["box_border_touch_ratio"],
        -summary["source_start_index"],
    ]
    return summary


def choose_candidate(candidates: list[dict[str, Any]], required_status: str = "pass") -> dict[str, Any]:
    if not candidates:
        return {
            "selection_status": "reject",
            "selection_reason": "no_dense_burst_candidates",
            "selected_candidate_index": None,
            "ground_truth_used": False,
            "manual_selection_used": False,
        }
    selected = max(candidates, key=lambda item: tuple(item["selection_rank"]))
    status = (
        "pass"
        if selected["prompt_quality_status"] == required_status
        else "reject"
    )
    return {
        "selection_status": status,
        "selection_reason": (
            "fixed_prediction_side_rank_selected_pass_candidate"
            if status == "pass"
            else "no_candidate_met_required_prompt_status"
        ),
        "selected_candidate_index": int(selected["candidate_index"]),
        "selected_source_start_index": int(selected["source_start_index"]),
        "selected_source_end_index": int(selected["source_end_index"]),
        "selected_prompt_quality_status": selected["prompt_quality_status"],
        "ground_truth_used": False,
        "manual_selection_used": False,
    }


def select_dense_burst(
    source_frames_dir: str | Path,
    source_fps: float,
    policy: dict[str, Any],
    detector_config: dict[str, Any],
    temporal_gate_config: dict[str, Any],
    prompt_config: dict[str, Any],
    output_dir: str | Path,
    runner: Callable[..., dict[str, Any]] = run_temporal_sam2_prompt_from_frames,
) -> dict[str, Any]:
    """Evaluate fixed candidate bursts and export the best one without reading GT."""
    source_dir = Path(source_frames_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if source_dir.name != "frames" or not source_dir.is_dir():
        raise ValueError("source_frames_dir must be an existing directory named frames")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dense-burst output: {output}")
    if source_fps <= 0.0:
        raise ValueError("source_fps must be positive")
    source_paths = sorted(source_dir.glob("frame_*.png"))
    if any(path.parent != source_dir for path in source_paths):
        raise ValueError("dense-burst selector only accepts direct frame children")
    burst_count = int(policy["burst_frame_count"])
    stride_frames = max(1, int(round(float(policy["candidate_stride_seconds"]) * source_fps)))
    starts = enumerate_candidate_starts(len(source_paths), burst_count, stride_frames)
    max_candidates = int(policy.get("max_candidate_count", len(starts)))
    if len(starts) > max_candidates:
        raise ValueError(
            f"candidate count {len(starts)} exceeds configured maximum {max_candidates}"
        )

    detector = dict(detector_config)
    detector["fps"] = float(source_fps)
    candidate_summaries: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    anchor = int(policy["anchor_frame_index"])
    with tempfile.TemporaryDirectory(prefix="water_agent_dense_burst_") as temporary:
        temp_root = Path(temporary)
        for candidate_index, start in enumerate(starts):
            candidate_dir = temp_root / f"candidate_{candidate_index:03d}" / "frames"
            _stage_frames(source_paths, start, burst_count, candidate_dir)
            result = runner(
                candidate_dir,
                candidate_dir / f"frame_{anchor:06d}.png",
                anchor,
                detector,
                temporal_gate_config,
                prompt_config,
            )
            candidate_results.append(result)
            candidate_summaries.append(
                summarize_candidate(
                    result,
                    candidate_index=candidate_index,
                    source_start_index=start,
                    source_end_index=start + burst_count - 1,
                    source_fps=source_fps,
                )
            )

    selection = choose_candidate(
        candidate_summaries,
        required_status=str(policy["required_prompt_status"]),
    )
    selected_index = selection["selected_candidate_index"]
    selected_result = candidate_results[selected_index] if selected_index is not None else None
    selected_summary = (
        candidate_summaries[selected_index] if selected_index is not None else None
    )
    output.mkdir(parents=True)
    if selected_summary is not None:
        selected_frames = output / "frames"
        _stage_frames(
            source_paths,
            int(selected_summary["source_start_index"]),
            burst_count,
            selected_frames,
        )
        prompt = dict(selected_result["prompt"])
        prompt["image_path"] = str(selected_frames / f"frame_{anchor:06d}.png")
        prompt["image_sha256"] = sha256_file(prompt["image_path"])
        prompt["dense_burst_selection_status"] = selection["selection_status"]
        prompt["dense_burst_source_start_index"] = int(
            selected_summary["source_start_index"]
        )
    else:
        prompt = None

    return {
        "schema_version": str(policy["schema_version"]),
        "algorithm_version": str(policy["algorithm_version"]),
        "source_frames_dir": str(source_dir),
        "source_frame_count": len(source_paths),
        "source_fps": float(source_fps),
        "burst_frame_count": burst_count,
        "candidate_stride_frames": stride_frames,
        "candidate_count": len(candidate_summaries),
        "anchor_frame_index": anchor,
        "selection": selection,
        "candidates": candidate_summaries,
        "selected_prompt": prompt,
        "selected_prompt_diagnostics": (
            selected_result["prompt_diagnostics"] if selected_result is not None else None
        ),
        "ground_truth_used": False,
        "manual_selection_used": False,
        "sam2_run_allowed": selection["selection_status"] == "pass",
    }
