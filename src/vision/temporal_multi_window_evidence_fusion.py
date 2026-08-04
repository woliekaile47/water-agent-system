#!/usr/bin/env python3
"""GT-free fusion of recurring temporal water evidence across fixed bursts."""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml

from src.vision.generate_temporal_sam2_prompt import generate_temporal_sam2_prompt
from src.vision.temporal_dense_burst_selection import enumerate_candidate_starts
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


def load_multi_window_fusion_policy(path: str | Path) -> dict[str, Any]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("multi-window fusion config root must be a mapping")
    forbidden = [
        key
        for key in _flatten_keys(document)
        if any(token in key for token in FORBIDDEN_CONFIG_TOKENS)
    ]
    if forbidden:
        raise ValueError(
            f"prediction-side multi-window config contains forbidden fields: {forbidden}"
        )
    policy = document.get("temporal_multi_window_evidence_fusion")
    if not isinstance(policy, dict):
        raise ValueError("temporal_multi_window_evidence_fusion mapping is required")
    if int(policy["minimum_eligible_window_count"]) < 1:
        raise ValueError("minimum_eligible_window_count must be positive")
    if int(policy["minimum_cluster_window_count"]) < 3:
        raise ValueError("minimum_cluster_window_count must be at least three")
    if float(policy["minimum_cluster_time_span_seconds"]) <= 0.0:
        raise ValueError("minimum_cluster_time_span_seconds must be positive")
    if not 0.0 <= float(policy["minimum_component_iou"]) <= 1.0:
        raise ValueError("minimum_component_iou must be between zero and one")
    if not 0.0 < float(policy["minimum_support_fraction"]) <= 1.0:
        raise ValueError("minimum_support_fraction must be in (0, 1]")
    if int(policy["minimum_support_window_count"]) < 1:
        raise ValueError("minimum_support_window_count must be positive")
    if int(policy["minimum_raw_support_window_count"]) < 1:
        raise ValueError("minimum_raw_support_window_count must be positive")
    if not 0.0 < float(policy["minimum_final_core_overlap_fraction"]) <= 1.0:
        raise ValueError("minimum_final_core_overlap_fraction must be in (0, 1]")
    if int(policy["support_tolerance_radius_pixels"]) < 0:
        raise ValueError("support_tolerance_radius_pixels must be non-negative")
    probability_weight = float(policy["probability_weight"])
    support_weight = float(policy["support_weight"])
    if not 0.0 <= probability_weight <= 1.0 or not 0.0 <= support_weight <= 1.0:
        raise ValueError("probability_weight and support_weight must be within [0, 1]")
    if not math.isclose(probability_weight + support_weight, 1.0, abs_tol=1e-9):
        raise ValueError("probability_weight and support_weight must sum to one")
    if int(policy["morphology_close_kernel"]) not in (1, 3, 5):
        raise ValueError("morphology_close_kernel must be one of 1, 3 or 5")
    unit_interval_fields = (
        "maximum_window_water_fraction",
        "window_ambiguous_component_area_ratio",
        "ambiguous_cluster_window_count_ratio",
        "minimum_fused_confidence",
        "fused_ambiguous_component_area_ratio",
        "unknown_consensus_fraction",
    )
    for field in unit_interval_fields:
        if not 0.0 <= float(policy[field]) <= 1.0:
            raise ValueError(f"{field} must be between zero and one")
    positive_integer_fields = (
        "minimum_window_component_area_pixels",
        "minimum_fused_component_area_pixels",
    )
    for field in positive_integer_fields:
        if int(policy[field]) < 1:
            raise ValueError(f"{field} must be positive")
    nonnegative_fields = (
        "maximum_centroid_distance_pixels",
        "cluster_overlap_tolerance_radius_pixels",
        "maximum_window_ambiguous_component_count",
        "maximum_fused_ambiguous_component_count",
    )
    for field in nonnegative_fields:
        if float(policy[field]) < 0.0:
            raise ValueError(f"{field} must be non-negative")
    if int(policy.get("required_source_frame_count", 0)) < 0:
        raise ValueError("required_source_frame_count must be non-negative")
    if policy.get("required_prompt_status") != "pass":
        raise ValueError("multi-window fusion must fail closed unless prompt status is pass")
    return policy


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, list[int], list[float]]:
    binary = np.asarray(mask, dtype=bool)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    records = [
        (int(stats[label, cv2.CC_STAT_AREA]), int(label))
        for label in range(1, count)
    ]
    records.sort(key=lambda item: (-item[0], item[1]))
    if not records:
        return np.zeros_like(binary), [], []
    selected = labels == records[0][1]
    areas = [item[0] for item in records]
    center = centroids[records[0][1]].astype(float).tolist()
    return selected, areas, center


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel) > 0


def _window_record(candidate: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    probability = np.asarray(candidate["probability"], dtype=np.float32)
    water = np.asarray(candidate["water_mask"], dtype=bool)
    unknown = np.asarray(candidate["unknown_mask"], dtype=bool)
    if probability.ndim != 2 or water.shape != probability.shape or unknown.shape != water.shape:
        raise ValueError("candidate probability, water and unknown arrays must be same-shape 2-D")
    exclusion: list[str] = []
    if not np.isfinite(probability).all():
        exclusion.append("nonfinite_probability")
    primary, component_areas, centroid = _largest_component(water & ~unknown)
    if not component_areas:
        exclusion.append("predicted_water_mask_empty")
    elif component_areas[0] < int(policy["minimum_window_component_area_pixels"]):
        exclusion.append("window_component_too_small")
    ambiguous = 0
    if component_areas:
        threshold = component_areas[0] * float(policy["window_ambiguous_component_area_ratio"])
        ambiguous = sum(area >= threshold for area in component_areas[1:])
        if ambiguous > int(policy["maximum_window_ambiguous_component_count"]):
            exclusion.append("ambiguous_window_components")

    gate_status = str(candidate.get("gate_status", "reject"))
    gate_reasons = [str(reason) for reason in candidate.get("gate_reasons", [])]
    if gate_status == "reject":
        exclusion.append("temporal_quality_gate_reject")
    elif gate_status == "partial":
        allowed = {str(reason) for reason in policy["allowed_partial_gate_reasons"]}
        if not gate_reasons or not set(gate_reasons).issubset(allowed):
            exclusion.append("partial_gate_reason_not_allowed")
        if candidate.get("observable_region_result_valid") is not True:
            exclusion.append("partial_gate_observable_region_invalid")
    elif gate_status != "pass":
        exclusion.append("temporal_quality_gate_unavailable")

    area_fraction = float(np.mean(primary))
    if area_fraction > float(policy["maximum_window_water_fraction"]):
        exclusion.append("window_water_fraction_too_large")
    return {
        "candidate_index": int(candidate["candidate_index"]),
        "source_start_index": int(candidate["source_start_index"]),
        "source_end_index": int(candidate["source_end_index"]),
        "source_start_seconds": float(candidate["source_start_seconds"]),
        "source_end_seconds": float(candidate["source_end_seconds"]),
        "gate_status": gate_status,
        "gate_reasons": gate_reasons,
        "component_count": len(component_areas),
        "component_areas_pixels": component_areas,
        "ambiguous_component_count": int(ambiguous),
        "primary_component_area_pixels": int(np.count_nonzero(primary)),
        "primary_component_centroid_xy": centroid,
        "water_mask_time_stability": float(candidate.get("water_mask_time_stability", 0.0)),
        "eligible_for_fusion": not exclusion,
        "fusion_exclusion_reasons": exclusion,
        "ground_truth_used": False,
        "_primary_component": primary,
        "_probability": probability,
        "_unknown": unknown,
    }


def _connected_clusters(records: list[dict[str, Any]], policy: dict[str, Any]) -> list[list[int]]:
    count = len(records)
    parents = list(range(count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    radius = int(policy["cluster_overlap_tolerance_radius_pixels"])
    minimum_iou = float(policy["minimum_component_iou"])
    maximum_distance = float(policy["maximum_centroid_distance_pixels"])
    dilated = [_dilate(item["_primary_component"], radius) for item in records]
    for left in range(count):
        for right in range(left + 1, count):
            iou = _mask_iou(records[left]["_primary_component"], records[right]["_primary_component"])
            center_left = np.asarray(records[left]["primary_component_centroid_xy"], dtype=float)
            center_right = np.asarray(records[right]["primary_component_centroid_xy"], dtype=float)
            distance = float(np.linalg.norm(center_left - center_right))
            tolerant_overlap = bool(np.any(dilated[left] & dilated[right]))
            if iou >= minimum_iou or (distance <= maximum_distance and tolerant_overlap):
                union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(find(index), []).append(index)
    return list(grouped.values())


def _cluster_summary(
    cluster_index: int,
    member_indices: list[int],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    members = [records[index] for index in member_indices]
    pairwise = [
        _mask_iou(members[left]["_primary_component"], members[right]["_primary_component"])
        for left in range(len(members))
        for right in range(left + 1, len(members))
    ]
    starts = [item["source_start_seconds"] for item in members]
    return {
        "cluster_index": int(cluster_index),
        "member_candidate_indices": [item["candidate_index"] for item in members],
        "member_source_start_indices": [item["source_start_index"] for item in members],
        "window_count": len(members),
        "time_span_seconds": float(max(starts) - min(starts)) if starts else 0.0,
        "pairwise_component_iou_median": float(np.median(pairwise)) if pairwise else 1.0,
        "earliest_source_start_index": min(item["source_start_index"] for item in members),
        "_member_indices": member_indices,
    }


def fuse_prediction_candidates(
    candidates: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray | int | None]]:
    """Fuse fixed prediction-side candidate windows without reading evaluation data."""
    if not candidates:
        raise ValueError("at least one candidate window is required")
    records = [_window_record(candidate, policy) for candidate in candidates]
    shapes = {record["_primary_component"].shape for record in records}
    if len(shapes) != 1:
        raise ValueError("candidate image shapes are inconsistent")
    shape = next(iter(shapes))
    eligible = [record for record in records if record["eligible_for_fusion"]]
    hard_reasons: list[str] = []
    if len(eligible) < int(policy["minimum_eligible_window_count"]):
        hard_reasons.append("insufficient_eligible_windows")

    cluster_records: list[dict[str, Any]] = []
    if eligible:
        for cluster_index, indices in enumerate(_connected_clusters(eligible, policy)):
            cluster_records.append(_cluster_summary(cluster_index, indices, eligible))
        cluster_records.sort(
            key=lambda item: (
                -item["window_count"],
                -item["time_span_seconds"],
                -item["pairwise_component_iou_median"],
                item["earliest_source_start_index"],
            )
        )
    if not cluster_records:
        hard_reasons.append("no_spatially_consistent_window_cluster")

    main_cluster = cluster_records[0] if cluster_records else None
    if main_cluster is not None:
        if len(cluster_records) > 1:
            ratio = cluster_records[1]["window_count"] / max(main_cluster["window_count"], 1)
            if ratio >= float(policy["ambiguous_cluster_window_count_ratio"]):
                hard_reasons.append("ambiguous_cross_window_components")

    raw_support_count = np.zeros(shape, dtype=np.uint16)
    support_count = np.zeros(shape, dtype=np.uint16)
    support_fraction = np.zeros(shape, dtype=np.float32)
    evidence_probability = np.zeros(shape, dtype=np.float32)
    fused_probability = np.zeros(shape, dtype=np.float32)
    fused_mask = np.zeros(shape, dtype=bool)
    fused_unknown = np.ones(shape, dtype=bool)
    representative_candidate_index: int | None = None
    required_support_count = int(policy["minimum_support_window_count"])
    fused_component_areas: list[int] = []
    fused_ambiguous_count = 0
    final_supporting_members: list[dict[str, Any]] = []
    final_supporting_count = 0
    final_supporting_time_span_seconds = 0.0

    if main_cluster is not None:
        members = [eligible[index] for index in main_cluster["_member_indices"]]
        support_radius = int(policy["support_tolerance_radius_pixels"])
        raw_component_stack = np.stack([
            np.asarray(item["_primary_component"], dtype=bool)
            for item in members
        ])
        component_stack = np.stack([
            _dilate(item["_primary_component"], support_radius)
            for item in members
        ])
        if support_radius > 0:
            size = 2 * support_radius + 1
            support_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (size, size)
            )
            probability_stack = np.stack([
                cv2.dilate(
                    (item["_probability"] * item["_primary_component"]).astype(np.float32),
                    support_kernel,
                )
                for item in members
            ])
        else:
            probability_stack = np.stack([
                item["_probability"] * item["_primary_component"]
                for item in members
            ])
        unknown_stack = np.stack([item["_unknown"] for item in members])
        raw_support_count = np.sum(raw_component_stack, axis=0, dtype=np.uint16)
        support_count = np.sum(component_stack, axis=0, dtype=np.uint16)
        support_fraction = support_count.astype(np.float32) / float(len(members))
        probability_sum = np.sum(probability_stack * component_stack, axis=0, dtype=np.float32)
        evidence_probability = np.divide(
            probability_sum,
            support_count,
            out=np.zeros(shape, dtype=np.float32),
            where=support_count > 0,
        )
        fused_probability = (
            float(policy["probability_weight"]) * evidence_probability
            + float(policy["support_weight"]) * support_fraction
        ).astype(np.float32)
        required_support_count = max(
            int(policy["minimum_support_window_count"]),
            int(math.ceil(float(policy["minimum_support_fraction"]) * len(members))),
        )
        raw_mask = (
            (support_count >= required_support_count)
            & (raw_support_count >= int(policy["minimum_raw_support_window_count"]))
            & (support_fraction >= float(policy["minimum_support_fraction"]))
            & (fused_probability >= float(policy["minimum_fused_confidence"]))
        )
        close_size = int(policy["morphology_close_kernel"])
        if close_size > 1:
            kernel = np.ones((close_size, close_size), dtype=np.uint8)
            raw_mask = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        # Morphology and tolerant voting may connect nearby raw observations, but they
        # must never invent a final water pixel that no original window predicted.
        raw_mask &= raw_support_count > 0
        fused_mask, fused_component_areas, _ = _largest_component(raw_mask)
        if not fused_component_areas:
            hard_reasons.append("fused_water_mask_empty")
        elif fused_component_areas[0] < int(policy["minimum_fused_component_area_pixels"]):
            hard_reasons.append("fused_component_too_small")
        if fused_component_areas:
            threshold = fused_component_areas[0] * float(policy["fused_ambiguous_component_area_ratio"])
            fused_ambiguous_count = sum(area >= threshold for area in fused_component_areas[1:])
            if fused_ambiguous_count > int(policy["maximum_fused_ambiguous_component_count"]):
                hard_reasons.append("ambiguous_fused_components")
        fused_unknown = (
            np.mean(unknown_stack, axis=0) >= float(policy["unknown_consensus_fraction"])
        ) & ~fused_mask

        fused_area = int(np.count_nonzero(fused_mask))
        if fused_area:
            minimum_overlap = float(policy["minimum_final_core_overlap_fraction"])
            for item in members:
                component = np.asarray(item["_primary_component"], dtype=bool)
                component_area = int(np.count_nonzero(component))
                overlap = int(np.count_nonzero(component & fused_mask))
                overlap_fraction = overlap / float(max(1, min(component_area, fused_area)))
                if overlap_fraction >= minimum_overlap:
                    final_supporting_members.append(item)
            final_supporting_count = len(final_supporting_members)
            starts = [item["source_start_seconds"] for item in final_supporting_members]
            final_supporting_time_span_seconds = (
                float(max(starts) - min(starts)) if starts else 0.0
            )
        if final_supporting_count < int(policy["minimum_cluster_window_count"]):
            hard_reasons.append("insufficient_cross_window_support")
        if final_supporting_time_span_seconds < float(policy["minimum_cluster_time_span_seconds"]):
            hard_reasons.append("cross_window_time_span_too_short")

        if final_supporting_members:
            representative = max(
                final_supporting_members,
                key=lambda item: (
                    _mask_iou(item["_primary_component"], fused_mask),
                    int(item["gate_status"] == "pass"),
                    item["water_mask_time_stability"],
                    -item["source_start_index"],
                ),
            )
            representative_candidate_index = int(representative["candidate_index"])

    hard_reasons = list(dict.fromkeys(hard_reasons))
    status = "pass" if not hard_reasons else "reject"
    public_records = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]
    public_clusters = [
        {key: value for key, value in cluster.items() if not key.startswith("_")}
        for cluster in cluster_records
    ]
    summary = {
        "fusion_status": status,
        "fusion_reasons": hard_reasons,
        "candidate_count": len(records),
        "eligible_window_count": len(eligible),
        "clusters": public_clusters,
        "selected_cluster_index": main_cluster["cluster_index"] if main_cluster else None,
        "preliminary_cluster_window_count": main_cluster["window_count"] if main_cluster else 0,
        "preliminary_cluster_time_span_seconds": main_cluster["time_span_seconds"] if main_cluster else 0.0,
        "selected_cluster_window_count": int(final_supporting_count),
        "selected_cluster_time_span_seconds": float(final_supporting_time_span_seconds),
        "final_core_supporting_candidate_indices": [
            int(item["candidate_index"]) for item in final_supporting_members
        ],
        "final_core_supporting_source_start_indices": [
            int(item["source_start_index"]) for item in final_supporting_members
        ],
        "required_support_count": int(required_support_count),
        "maximum_support_count": int(np.max(support_count)),
        "fused_component_areas_pixels": fused_component_areas,
        "fused_ambiguous_component_count": int(fused_ambiguous_count),
        "representative_candidate_index": representative_candidate_index,
        "candidates": public_records,
        "ground_truth_used": False,
        "manual_selection_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
    }
    artifacts: dict[str, np.ndarray | int | None] = {
        "raw_support_count": raw_support_count,
        "support_count": support_count,
        "support_fraction": support_fraction,
        "evidence_probability": evidence_probability,
        "fused_probability": fused_probability,
        "fused_water_mask": fused_mask,
        "fused_unknown_mask": fused_unknown,
        "representative_candidate_index": representative_candidate_index,
    }
    return summary, artifacts


def _stage_frames(source_paths: list[Path], start: int, count: int, frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=False)
    for local_index, source in enumerate(source_paths[start : start + count]):
        shutil.copy2(source, frames_dir / f"frame_{local_index:06d}.png")


def run_temporal_multi_window_fusion(
    source_frames_dir: str | Path,
    source_fps: float,
    window_policy: dict[str, Any],
    fusion_policy: dict[str, Any],
    detector_config: dict[str, Any],
    temporal_gate_config: dict[str, Any],
    prompt_config: dict[str, Any],
    output_dir: str | Path,
    runner: Callable[..., dict[str, Any]] = run_temporal_sam2_prompt_from_frames,
) -> tuple[dict[str, Any], dict[str, np.ndarray | int | None]]:
    """Run fixed bursts, fuse recurring evidence, and export one frozen prompt."""
    source_dir = Path(source_frames_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if source_dir.name != "frames" or not source_dir.is_dir():
        raise ValueError("source_frames_dir must be an existing directory named frames")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite multi-window output: {output}")
    if source_fps <= 0.0:
        raise ValueError("source_fps must be positive")
    source_paths = sorted(source_dir.glob("frame_*.png"))
    frame_pattern = re.compile(r"^frame_(\d+)\.png$")
    frame_indices: list[int] = []
    for path in source_paths:
        match = frame_pattern.match(path.name)
        if match is None:
            raise ValueError(f"unexpected source frame name: {path.name}")
        frame_indices.append(int(match.group(1)))
    if frame_indices != list(range(len(source_paths))):
        raise ValueError("source frame indices must be contiguous and start at zero")
    required_source_count = int(fusion_policy.get("required_source_frame_count", 0))
    if required_source_count and len(source_paths) != required_source_count:
        raise ValueError(
            f"expected {required_source_count} source frames, found {len(source_paths)}"
        )
    burst_count = int(window_policy["burst_frame_count"])
    anchor = int(window_policy["anchor_frame_index"])
    stride_frames = max(1, int(round(float(window_policy["candidate_stride_seconds"]) * source_fps)))
    starts = enumerate_candidate_starts(len(source_paths), burst_count, stride_frames)
    if len(starts) > int(window_policy.get("max_candidate_count", len(starts))):
        raise ValueError("candidate count exceeds configured maximum")
    detector = dict(detector_config)
    detector["fps"] = float(source_fps)
    fusion_candidates: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="water_agent_multi_window_") as temporary:
        root = Path(temporary)
        for candidate_index, start in enumerate(starts):
            frames = root / f"candidate_{candidate_index:03d}" / "frames"
            _stage_frames(source_paths, start, burst_count, frames)
            result = runner(
                frames,
                frames / f"frame_{anchor:06d}.png",
                anchor,
                detector,
                temporal_gate_config,
                prompt_config,
            )
            prediction = result["prediction"]
            gate = result["temporal_quality_gate"]
            fusion_candidates.append({
                "candidate_index": candidate_index,
                "source_start_index": start,
                "source_end_index": start + burst_count - 1,
                "source_start_seconds": start / source_fps,
                "source_end_seconds": (start + burst_count - 1) / source_fps,
                "probability": prediction["evidence"]["predicted_water_probability"],
                "water_mask": prediction["evidence"]["predicted_water_mask"],
                "unknown_mask": prediction["evidence"]["predicted_unknown_mask"],
                "gate_status": gate.get("status", "reject"),
                "gate_reasons": gate.get("reasons", []),
                "observable_region_result_valid": gate.get("observable_region_result_valid"),
                "water_mask_time_stability": prediction.get("water_mask_time_stability", 0.0),
            })

    fusion, artifacts = fuse_prediction_candidates(fusion_candidates, fusion_policy)
    representative_index = artifacts["representative_candidate_index"]
    output.mkdir(parents=True)
    prompt: dict[str, Any] | None = None
    prompt_diagnostics: dict[str, Any] | None = None
    selected_start: int | None = None
    if representative_index is not None:
        selected_start = starts[int(representative_index)]
        selected_frames = output / "frames"
        _stage_frames(source_paths, selected_start, burst_count, selected_frames)
        image_path = selected_frames / f"frame_{anchor:06d}.png"
        fusion_gate = {
            "status": fusion["fusion_status"],
            "reasons": fusion["fusion_reasons"],
            "observable_region_result_valid": fusion["fusion_status"] == "pass",
            "ground_truth_used": False,
        }
        prompt, prompt_diagnostics = generate_temporal_sam2_prompt(
            np.asarray(artifacts["fused_probability"], dtype=np.float32),
            np.asarray(artifacts["fused_water_mask"], dtype=bool),
            np.asarray(artifacts["fused_unknown_mask"], dtype=bool),
            [],
            fusion_gate,
            prompt_config,
            temporal_support_fraction=np.asarray(artifacts["support_fraction"], dtype=np.float32),
            image_path=str(image_path),
            image_sha256=sha256_file(image_path),
            frame_index=anchor,
        )
        prompt["prompt_source"] = "temporal_multi_window_evidence_fusion_v1"
        prompt["multi_window_fusion_status"] = fusion["fusion_status"]
        prompt["supporting_window_count"] = fusion["selected_cluster_window_count"]
        prompt["supporting_time_span_seconds"] = fusion["selected_cluster_time_span_seconds"]
        prompt["representative_source_start_index"] = selected_start
        prompt["ground_truth_used"] = False
        prompt["authoritative"] = False
        prompt["eligible_for_downstream"] = False
        prompt_diagnostics["multi_window_fusion_status"] = fusion["fusion_status"]
        prompt_diagnostics["ground_truth_used"] = False
        prompt_diagnostics["eligible_for_downstream"] = False

    sam2_allowed = bool(
        fusion["fusion_status"] == "pass"
        and prompt is not None
        and prompt.get("prompt_quality_status") == fusion_policy["required_prompt_status"]
    )
    result = {
        "schema_version": str(fusion_policy["schema_version"]),
        "algorithm_version": str(fusion_policy["algorithm_version"]),
        "source_frames_dir": str(source_dir),
        "source_frame_count": len(source_paths),
        "source_fps": float(source_fps),
        "burst_frame_count": burst_count,
        "candidate_stride_frames": stride_frames,
        "anchor_frame_index": anchor,
        "fusion": fusion,
        "selected_source_start_index": selected_start,
        "selected_prompt": prompt,
        "selected_prompt_diagnostics": prompt_diagnostics,
        "sam2_run_allowed": sam2_allowed,
        "ground_truth_used": False,
        "manual_selection_used": False,
        "authoritative": False,
        "eligible_for_downstream": False,
    }
    return result, artifacts
