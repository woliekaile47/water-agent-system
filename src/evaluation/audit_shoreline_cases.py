#!/usr/bin/env python3
"""Read-only visual audit of selected Phase 2D-A shoreline cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.fusion.water_surface_aware_mask_to_dem import _nearest_boundary_distances
from src.hydrology.invert_boundary_waterline_depth import extract_boundary_mask
from src.integration.unknown_aware_geometry import build_trusted_shoreline


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def case_key(record: dict[str, Any]) -> str:
    return f"{record['case_id']}__{record['rain_level']}__seed_{record['seed']}"


def select_audit_cases(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: (row["case_id"], row["rain_level"], row["seed"]))
    groups = {
        "boundary_p95_above_10_px": [row for row in ordered if row["boundary_reprojection_p95_px"] > 10],
        "boundary_p95_above_3_through_4_px": [
            row for row in ordered if 3 < row["boundary_reprojection_p95_px"] <= 4
        ],
        "camera_iou_at_least_0_8_geometry_reject": [
            row for row in ordered
            if row["camera_mask_iou"] >= 0.8 and row.get("geometry_gate_status") == "reject"
        ],
    }
    reasons: dict[str, list[str]] = {}
    lookup: dict[str, dict[str, Any]] = {}
    for group_name, group_records in groups.items():
        for record in group_records:
            key = case_key(record)
            lookup[key] = record
            reasons.setdefault(key, []).append(group_name)
    selected = []
    for key in sorted(lookup):
        selected.append({
            "case_key": key,
            "case_id": lookup[key]["case_id"],
            "rain_level": lookup[key]["rain_level"],
            "seed": lookup[key]["seed"],
            "selection_groups": reasons[key],
        })
    return {
        "selection_policy": {
            "boundary_p95_above_10_px": "p95 > 10",
            "boundary_p95_above_3_through_4_px": "3 < p95 <= 4",
            "camera_iou_at_least_0_8_geometry_reject": "camera IoU >= 0.8 and geometry gate reject",
            "duplicates_allowed_between_groups": True,
            "union_deduplicated_for_audit": True,
        },
        "group_counts": {name: len(group) for name, group in groups.items()},
        "groups": {name: [case_key(row) for row in group] for name, group in groups.items()},
        "selected_unique_count": len(selected),
        "selected_cases": selected,
    }


def _component_metrics(mask: np.ndarray) -> dict[str, Any]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, dtype=np.uint8), connectivity=8)
    sizes = sorted((int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)), reverse=True)
    return {
        "component_count": len(sizes),
        "component_sizes_px": sizes,
        "largest_component_size_px": sizes[0] if sizes else 0,
        "non_largest_component_count": max(0, len(sizes) - 1),
        "components_at_most_100px_count": sum(size <= 100 for size in sizes[1:]),
    }


def _distance_map(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = _nearest_boundary_distances(source, target)
    result = np.full(source.shape, np.nan, dtype=np.float32)
    points = np.column_stack(np.where(source))
    if distances.size:
        result[points[:, 0], points[:, 1]] = distances.astype(np.float32)
    return result, distances


def boundary_spatial_diagnostics(
    observed_water: np.ndarray, unknown: np.ndarray, reprojected_water: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    observed = np.asarray(observed_water, dtype=bool)
    unknown_mask = np.asarray(unknown, dtype=bool)
    reprojected = np.asarray(reprojected_water, dtype=bool) & ~unknown_mask
    observed_boundary, _, _ = build_trusted_shoreline(observed, unknown_mask)
    reprojected_boundary, _, _ = build_trusted_shoreline(reprojected, unknown_mask)
    forward_map, forward = _distance_map(observed_boundary, reprojected_boundary)
    reverse_map, reverse = _distance_map(reprojected_boundary, observed_boundary)
    symmetric = np.concatenate((forward, reverse)) if forward.size and reverse.size else np.asarray([])
    tail_mask = (np.nan_to_num(forward_map, nan=-1) > 10) | (np.nan_to_num(reverse_map, nan=-1) > 10)
    tail_count = int(np.count_nonzero(tail_mask))
    tail_bbox_fraction = 0.0
    if tail_count:
        rows, columns = np.where(tail_mask)
        tail_bbox_fraction = float(
            (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1) / tail_mask.size
        )
    raw_boundary = extract_boundary_mask(observed)
    unknown_adjacent = cv2.dilate(unknown_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    unknown_touch_count = int(np.count_nonzero(raw_boundary & unknown_adjacent))
    raw_boundary_count = int(np.count_nonzero(raw_boundary))
    metrics = {
        "trusted_observed_boundary_pixel_count": int(np.count_nonzero(observed_boundary)),
        "trusted_reprojected_boundary_pixel_count": int(np.count_nonzero(reprojected_boundary)),
        "boundary_distance_mean_px": float(np.mean(symmetric)) if symmetric.size else None,
        "boundary_distance_p50_px": float(np.percentile(symmetric, 50)) if symmetric.size else None,
        "boundary_distance_p95_px": float(np.percentile(symmetric, 95)) if symmetric.size else None,
        "boundary_distance_above_3_fraction": float(np.mean(symmetric > 3)) if symmetric.size else None,
        "boundary_distance_above_10_fraction": float(np.mean(symmetric > 10)) if symmetric.size else None,
        "boundary_tail_above_10_pixel_count": tail_count,
        "boundary_tail_above_10_bbox_image_fraction": tail_bbox_fraction,
        "unknown_touching_raw_water_boundary_pixel_count": unknown_touch_count,
        "unknown_touching_raw_water_boundary_fraction": float(unknown_touch_count / max(1, raw_boundary_count)),
        "observed_components": _component_metrics(observed),
        "reprojected_components": _component_metrics(reprojected),
    }
    arrays = {
        "observed_boundary": observed_boundary,
        "reprojected_boundary": reprojected_boundary,
        "forward_distance_map": forward_map,
        "reverse_distance_map": reverse_map,
        "tail_mask": tail_mask,
    }
    return metrics, arrays


def temporal_curve_diagnostics(temporal: dict[str, Any]) -> dict[str, Any]:
    frame_count = int((temporal.get("loader") or {}).get("frame_count", 0))
    candidate_counts = list((temporal.get("candidates") or {}).get("candidate_count_by_frame", []))
    classifications = {
        item["track_id"]: item.get("classification") for item in temporal.get("classifications", [])
    }
    water_track_area_proxy = [0.0] * frame_count
    for track in temporal.get("tracks", []):
        if classifications.get(track.get("track_id")) != "water_ripple":
            continue
        for observation in track.get("observations", []):
            frame_index = int(observation.get("frame_index", -1))
            if 0 <= frame_index < frame_count:
                water_track_area_proxy[frame_index] += float(observation.get("area", 0))
    return {
        "frame_count": frame_count,
        "candidate_count_by_frame": candidate_counts,
        "candidate_count_curve_available": len(candidate_counts) == frame_count and frame_count > 0,
        "classified_water_track_observation_area_proxy_by_frame": water_track_area_proxy,
        "classified_water_track_area_proxy_available": frame_count > 0,
        "classified_water_track_area_proxy_semantics": "sum_of_track_observation_areas_may_overlap_not_a_water_mask_area",
        "temporal_water_mask_area_curve_available": False,
        "temporal_water_mask_area_curve_unavailable_reason": "per_frame_water_masks_not_saved_in_existing_artifacts",
        "temporal_shoreline_stability_curve_available": False,
        "temporal_shoreline_stability_curve_unavailable_reason": "per_frame_shorelines_not_saved_in_existing_artifacts",
        "water_mask_time_stability_scalar": temporal.get("water_mask_time_stability"),
    }


def audit_case(
    prediction_dir: str | Path, geometry_record: dict[str, Any], selection_groups: list[str],
) -> dict[str, Any]:
    prediction = Path(prediction_dir).expanduser().resolve()
    required = [
        "predicted_camera_water_mask.png", "predicted_camera_unknown_mask.png",
        "reprojected_camera_mask.png", "temporal_diagnostics.json", "prediction_manifest.json",
    ]
    missing = [name for name in required if not (prediction / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing existing audit artifacts: {missing}")
    manifest = _read_json(prediction / "prediction_manifest.json")
    if manifest.get("data_role") != "prediction" or manifest.get("ground_truth_or_metadata_read_during_prediction") is not False:
        raise ValueError("Prediction manifest does not prove GT-isolated prediction")
    observed = _read_mask(prediction / "predicted_camera_water_mask.png")
    unknown = _read_mask(prediction / "predicted_camera_unknown_mask.png")
    reprojected = _read_mask(prediction / "reprojected_camera_mask.png")
    spatial, _ = boundary_spatial_diagnostics(observed, unknown, reprojected)
    temporal = temporal_curve_diagnostics(_read_json(prediction / "temporal_diagnostics.json"))
    known_reprojected = reprojected & ~unknown
    intersection = int(np.count_nonzero(observed & known_reprojected))
    return {
        "case_key": case_key(geometry_record),
        "case_id": geometry_record["case_id"], "nominal_depth_cm": geometry_record["nominal_depth_cm"],
        "rain_level": geometry_record["rain_level"], "seed": geometry_record["seed"],
        "selection_groups": selection_groups,
        "prediction_dir": str(prediction),
        "diagnostic_png": f"cases/{case_key(geometry_record)}.png",
        "camera_mask_iou": geometry_record["camera_mask_iou"],
        "camera_reprojection_iou": geometry_record["camera_reprojection_iou"],
        "boundary_reprojection_p95_px": geometry_record["boundary_reprojection_p95_px"],
        "water_level_absolute_error_m": geometry_record["water_level_absolute_error_m"],
        "depth_mae_m": geometry_record["depth_mae_m"],
        "visual_gate_status": geometry_record["visual_gate_status"],
        "geometry_reject_reasons": geometry_record["geometry_reject_reasons"],
        "observed_water_pixel_count": int(np.count_nonzero(observed)),
        "reprojected_known_water_pixel_count": int(np.count_nonzero(known_reprojected)),
        "observed_to_reprojected_area_ratio": float(np.count_nonzero(known_reprojected) / max(1, np.count_nonzero(observed))),
        "known_region_mask_intersection_pixel_count": intersection,
        "spatial": spatial,
        "temporal": temporal,
        "prediction_rerun": False,
        "algorithm_or_threshold_modified": False,
        "eligible_for_downstream": False,
    }


CSV_FIELDS = [
    "case_id", "nominal_depth_cm", "rain_level", "seed", "selection_groups",
    "camera_mask_iou", "camera_reprojection_iou", "boundary_reprojection_p95_px",
    "water_level_absolute_error_m", "depth_mae_m", "visual_gate_status", "geometry_reject_reasons",
    "observed_water_pixel_count", "reprojected_known_water_pixel_count", "observed_to_reprojected_area_ratio",
    "observed_component_count", "reprojected_component_count", "observed_small_component_count",
    "reprojected_small_component_count", "boundary_distance_above_3_fraction",
    "boundary_distance_above_10_fraction", "boundary_tail_above_10_bbox_image_fraction",
    "unknown_touching_raw_water_boundary_fraction", "water_mask_time_stability_scalar",
    "temporal_water_mask_area_curve_available", "temporal_shoreline_stability_curve_available",
]


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    spatial, temporal = record["spatial"], record["temporal"]
    return {
        "case_id": record["case_id"], "nominal_depth_cm": record["nominal_depth_cm"],
        "rain_level": record["rain_level"], "seed": record["seed"],
        "selection_groups": ";".join(record["selection_groups"]),
        "camera_mask_iou": record["camera_mask_iou"],
        "camera_reprojection_iou": record["camera_reprojection_iou"],
        "boundary_reprojection_p95_px": record["boundary_reprojection_p95_px"],
        "water_level_absolute_error_m": record["water_level_absolute_error_m"],
        "depth_mae_m": record["depth_mae_m"], "visual_gate_status": record["visual_gate_status"],
        "geometry_reject_reasons": ";".join(record["geometry_reject_reasons"]),
        "observed_water_pixel_count": record["observed_water_pixel_count"],
        "reprojected_known_water_pixel_count": record["reprojected_known_water_pixel_count"],
        "observed_to_reprojected_area_ratio": record["observed_to_reprojected_area_ratio"],
        "observed_component_count": spatial["observed_components"]["component_count"],
        "reprojected_component_count": spatial["reprojected_components"]["component_count"],
        "observed_small_component_count": spatial["observed_components"]["components_at_most_100px_count"],
        "reprojected_small_component_count": spatial["reprojected_components"]["components_at_most_100px_count"],
        "boundary_distance_above_3_fraction": spatial["boundary_distance_above_3_fraction"],
        "boundary_distance_above_10_fraction": spatial["boundary_distance_above_10_fraction"],
        "boundary_tail_above_10_bbox_image_fraction": spatial["boundary_tail_above_10_bbox_image_fraction"],
        "unknown_touching_raw_water_boundary_fraction": spatial["unknown_touching_raw_water_boundary_fraction"],
        "water_mask_time_stability_scalar": temporal["water_mask_time_stability_scalar"],
        "temporal_water_mask_area_curve_available": temporal["temporal_water_mask_area_curve_available"],
        "temporal_shoreline_stability_curve_available": temporal["temporal_shoreline_stability_curve_available"],
    }


def build_audit_summary(selection: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    extreme = [case for case in cases if case["boundary_reprojection_p95_px"] > 10]
    near = [case for case in cases if 3 < case["boundary_reprojection_p95_px"] <= 4]
    high_iou = [case for case in cases if case["camera_mask_iou"] >= 0.8]
    return {
        "data_role": "read_only_shoreline_case_audit",
        "selection": selection,
        "audited_unique_case_count": len(cases),
        "case_group_counts_after_overlap": {
            "extreme_p95": len(extreme), "near_threshold_p95": len(near), "high_camera_iou": len(high_iou),
        },
        "cases_with_multiple_observed_components": sum(case["spatial"]["observed_components"]["component_count"] > 1 for case in cases),
        "cases_with_small_non_largest_observed_components": sum(case["spatial"]["observed_components"]["components_at_most_100px_count"] > 0 for case in cases),
        "cases_with_unknown_touching_raw_shoreline": sum(case["spatial"]["unknown_touching_raw_water_boundary_pixel_count"] > 0 for case in cases),
        "temporal_water_mask_area_curve_available_count": sum(case["temporal"]["temporal_water_mask_area_curve_available"] for case in cases),
        "temporal_shoreline_stability_curve_available_count": sum(case["temporal"]["temporal_shoreline_stability_curve_available"] for case in cases),
        "classified_water_track_area_proxy_available_count": sum(case["temporal"]["classified_water_track_area_proxy_available"] for case in cases),
        "cases": cases,
        "extreme_cases": extreme,
        "near_threshold_cases": near,
        "high_camera_iou_cases": high_iou,
        "prediction_rerun": False,
        "algorithm_threshold_gate_or_gt_modified": False,
        "eligible_for_downstream": False,
    }


def write_audit_outputs(output_root: str | Path, selection: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases").mkdir(parents=True, exist_ok=True)
    summary = build_audit_summary(selection, cases)
    (output / "case_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "case_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "case_audit_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_flatten(case) for case in cases)
    write_figures_compatible(output)
    return summary


def write_figures_compatible(output: Path) -> None:
    if int(np.__version__.split(".", 1)[0]) >= 2 and os.environ.get("PHASE2DB2A_MPL_COMPAT") != "1":
        environment = dict(os.environ)
        environment.update({"PYTHONNOUSERSITE": "1", "PHASE2DB2A_MPL_COMPAT": "1"})
        subprocess.run(
            ["/usr/bin/python3", "-m", "src.evaluation.audit_shoreline_cases", "--render-existing", str(output)],
            check=True, env=environment,
        )
        return
    write_case_figures(output)


def write_case_figures(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _read_json(output / "case_audit_summary.json")
    for case in summary["cases"]:
        target = output / case["diagnostic_png"]
        prediction = Path(case["prediction_dir"])
        observed = _read_mask(prediction / "predicted_camera_water_mask.png")
        unknown = _read_mask(prediction / "predicted_camera_unknown_mask.png")
        reprojected = _read_mask(prediction / "reprojected_camera_mask.png") & ~unknown
        _, arrays = boundary_spatial_diagnostics(observed, unknown, reprojected)
        overlay = np.zeros((*observed.shape, 3), dtype=np.uint8)
        overlay[unknown] = (55, 55, 55)
        overlay[arrays["observed_boundary"]] = (255, 70, 70)
        overlay[arrays["reprojected_boundary"]] = (50, 220, 255)
        overlap = arrays["observed_boundary"] & arrays["reprojected_boundary"]
        overlay[overlap] = (255, 255, 50)
        distance_map = np.fmax(arrays["forward_distance_map"], arrays["reverse_distance_map"])

        figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        axes[0, 0].imshow(observed, cmap="Blues", vmin=0, vmax=1); axes[0, 0].set_title("Predicted Camera water mask")
        axes[0, 1].imshow(unknown, cmap="gray", vmin=0, vmax=1); axes[0, 1].set_title("Unknown mask")
        axes[0, 2].imshow(reprojected, cmap="Greens", vmin=0, vmax=1); axes[0, 2].set_title("Reprojected Camera mask")
        axes[1, 0].imshow(overlay); axes[1, 0].set_title("Boundaries: observed red / reprojected cyan")
        image = axes[1, 1].imshow(np.ma.masked_invalid(distance_map), cmap="magma", vmin=0)
        axes[1, 1].set_title("Nearest-boundary distance (px)"); figure.colorbar(image, ax=axes[1, 1], fraction=0.046)
        temporal = case["temporal"]
        if temporal["candidate_count_curve_available"]:
            axes[1, 2].plot(temporal["candidate_count_by_frame"], color="tab:blue", label="candidate count")
        axes[1, 2].plot(
            temporal["classified_water_track_observation_area_proxy_by_frame"],
            color="tab:orange", alpha=0.8, label="water-track area proxy",
        )
        axes[1, 2].set_title(
            "Temporal proxies\nactual mask-area / shoreline curves unavailable\n"
            f"stability scalar={temporal['water_mask_time_stability_scalar']:.3f}"
        )
        axes[1, 2].set_xlabel("Frame index"); axes[1, 2].legend(fontsize=8)
        for axis in axes.flat[:5]:
            axis.axis("off")
        figure.suptitle(
            f"{case['case_id']} / {case['rain_level']} / seed_{case['seed']} | "
            f"Camera IoU={case['camera_mask_iou']:.3f}, boundary P95={case['boundary_reprojection_p95_px']:.3f}px"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        figure.savefig(target, dpi=150)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-existing")
    args = parser.parse_args()
    if not args.render_existing:
        parser.error("--render-existing is required when invoking this module directly")
    write_case_figures(Path(args.render_existing).expanduser().resolve())


if __name__ == "__main__":
    main()
