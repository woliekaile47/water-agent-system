#!/usr/bin/env python3
"""Read-only Phase 2D-B-1 diagnostics for existing reprojection artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

from src.fusion.water_surface_aware_mask_to_dem import _nearest_boundary_distances
from src.integration.unknown_aware_geometry import build_trusted_shoreline


BOUNDARY_GATE_THRESHOLD_PX = 3.0
PHASE_RESPONSE_REPORTING_LEVEL = 0.20


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def _finite(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if value is not None and np.isfinite(value)]


def _mean(values: Iterable[Any]) -> float | None:
    finite = _finite(values)
    return None if not finite else float(np.mean(finite))


def _median(values: Iterable[Any]) -> float | None:
    finite = _finite(values)
    return None if not finite else float(np.median(finite))


def _percentile(values: Iterable[Any], percentile: float) -> float | None:
    finite = _finite(values)
    return None if not finite else float(np.percentile(finite, percentile))


def _pearson(left: Iterable[Any], right: Iterable[Any]) -> float | None:
    pairs = [
        (float(x), float(y)) for x, y in zip(left, right)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)
    ]
    if len(pairs) < 2:
        return None
    x_values, y_values = np.asarray(pairs, dtype=np.float64).T
    if np.std(x_values) == 0 or np.std(y_values) == 0:
        return None
    return float(np.corrcoef(x_values, y_values)[0, 1])


def boundary_distance_diagnostics(
    observed_water: np.ndarray,
    unknown: np.ndarray,
    reprojected_water: np.ndarray,
) -> dict[str, Any]:
    """Recompute the frozen unknown-aware symmetric boundary-distance domain."""
    observed = np.asarray(observed_water, dtype=bool)
    unknown_mask = np.asarray(unknown, dtype=bool)
    reprojected = np.asarray(reprojected_water, dtype=bool)
    if observed.shape != unknown_mask.shape or observed.shape != reprojected.shape:
        raise ValueError("observed, unknown, and reprojected masks must have identical shapes")
    observed_boundary, _, _ = build_trusted_shoreline(observed, unknown_mask)
    predicted_boundary, _, _ = build_trusted_shoreline(reprojected & ~unknown_mask, unknown_mask)
    forward = _nearest_boundary_distances(observed_boundary, predicted_boundary)
    reverse = _nearest_boundary_distances(predicted_boundary, observed_boundary)
    symmetric = np.concatenate((forward, reverse)) if forward.size and reverse.size else np.asarray([])
    return {
        "boundary_reprojection_mean_px": _mean(symmetric),
        "boundary_reprojection_p50_px": _percentile(symmetric, 50),
        "boundary_reprojection_p95_px": _percentile(symmetric, 95),
        "observed_to_reprojected_mean_px": _mean(forward),
        "observed_to_reprojected_p95_px": _percentile(forward, 95),
        "reprojected_to_observed_mean_px": _mean(reverse),
        "reprojected_to_observed_p95_px": _percentile(reverse, 95),
        "trusted_observed_boundary_pixel_count": int(np.count_nonzero(observed_boundary)),
        "trusted_reprojected_boundary_pixel_count": int(np.count_nonzero(predicted_boundary)),
        # Nearest-neighbour boundary matches are many-to-one and not physical correspondences.
        "signed_dx_mean_px": None,
        "signed_dy_mean_px": None,
        "signed_dx_median_px": None,
        "signed_dy_median_px": None,
        "signed_offset_status": "unavailable_no_reliable_one_to_one_boundary_correspondence",
    }


def phase_correlation_translation(
    observed_water: np.ndarray,
    unknown: np.ndarray,
    reprojected_water: np.ndarray,
) -> dict[str, Any]:
    """Estimate one exploratory global translation; it is not point correspondence."""
    known = ~np.asarray(unknown, dtype=bool)
    observed = (np.asarray(observed_water, dtype=bool) & known).astype(np.float32)
    reprojected = (np.asarray(reprojected_water, dtype=bool) & known).astype(np.float32)
    if not np.any(observed) or not np.any(reprojected):
        return {
            "phase_alignment_dx_px": None, "phase_alignment_dy_px": None,
            "phase_alignment_response": None, "phase_alignment_reporting_level_met": False,
            "phase_alignment_semantics": "unavailable_empty_mask",
        }
    height, width = observed.shape
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(reprojected, observed, window)
    return {
        # Shift to apply to the reprojected mask to align it with the observed mask.
        "phase_alignment_dx_px": float(dx),
        "phase_alignment_dy_px": float(dy),
        "phase_alignment_response": float(response),
        "phase_alignment_reporting_level_met": bool(response >= PHASE_RESPONSE_REPORTING_LEVEL),
        "phase_alignment_semantics": "exploratory_global_translation_not_boundary_correspondence",
    }


def analyze_sequence_artifacts(
    prediction_dir: str | Path,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    output = Path(prediction_dir).expanduser().resolve()
    if evaluation.get("is_dry"):
        raise ValueError("Phase 2D-B-1 geometry diagnostics only accept water sequences")
    required = [
        "prediction_manifest.json", "predicted_camera_water_mask.png",
        "predicted_camera_unknown_mask.png", "reprojected_camera_mask.png",
        "self_consistency.json", "geometry_quality_gate.json", "visual_quality_gate.json",
        "shoreline_intersections.json", "ray_intersection_diagnostics.json",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing existing prediction artifacts: {missing}")
    manifest = _read_json(output / "prediction_manifest.json")
    if manifest.get("data_role") != "prediction":
        raise ValueError("Artifact manifest is not a prediction manifest")
    if manifest.get("ground_truth_or_metadata_read_during_prediction") is not False:
        raise ValueError("Artifact manifest does not prove prediction/GT isolation")
    observed = _read_mask(output / "predicted_camera_water_mask.png")
    unknown = _read_mask(output / "predicted_camera_unknown_mask.png")
    reprojected = _read_mask(output / "reprojected_camera_mask.png")
    self_consistency = _read_json(output / "self_consistency.json")
    geometry_gate = _read_json(output / "geometry_quality_gate.json")
    visual_gate = _read_json(output / "visual_quality_gate.json")
    shoreline_document = _read_json(output / "shoreline_intersections.json")
    ray_diagnostics = _read_json(output / "ray_intersection_diagnostics.json")
    absolute_ray_residuals = _finite(
        abs(item["residual_m"]) for item in shoreline_document.get("intersections", [])
        if item.get("residual_m") is not None
    )
    boundary = boundary_distance_diagnostics(observed, unknown, reprojected)
    full_domain_boundary = boundary_distance_diagnostics(observed, np.zeros_like(unknown), reprojected)
    phase = phase_correlation_translation(observed, unknown, reprojected)
    stored_p95 = self_consistency.get("boundary_reprojection_p95_px")
    recomputed_p95 = boundary["boundary_reprojection_p95_px"]
    p95_delta = None if stored_p95 is None or recomputed_p95 is None else abs(float(stored_p95) - recomputed_p95)
    full_p95 = full_domain_boundary["boundary_reprojection_p95_px"]
    domain_p95_delta = None if recomputed_p95 is None or full_p95 is None else recomputed_p95 - full_p95
    full_intersection = int(np.count_nonzero(observed & reprojected))
    full_union = int(np.count_nonzero(observed | reprojected))
    return {
        "data_role": "diagnostic_from_existing_prediction_and_evaluation_artifacts",
        "case_id": evaluation["case_id"],
        "nominal_depth_cm": evaluation["nominal_depth_cm"],
        "rain_level": evaluation["rain_level"],
        "seed": evaluation["seed"],
        "camera_mask_iou": evaluation["camera_mask"]["whole_image_iou"],
        "known_region_iou": evaluation["camera_mask"]["known_region_iou"],
        "boundary_reprojection_p50_px": boundary["boundary_reprojection_p50_px"],
        "boundary_reprojection_p95_px": recomputed_p95,
        "stored_boundary_reprojection_p95_px": stored_p95,
        "stored_recomputed_p95_absolute_delta_px": p95_delta,
        "boundary_reprojection_mean_px": boundary["boundary_reprojection_mean_px"],
        "full_image_boundary_reprojection_p95_px": full_p95,
        "unknown_aware_minus_full_image_p95_px": domain_p95_delta,
        "camera_reprojection_iou": self_consistency.get("camera_reprojection_iou"),
        "full_image_camera_reprojection_iou": float(full_intersection / max(1, full_union)),
        "water_level_absolute_error_m": (evaluation.get("water_level") or {}).get("water_level_absolute_error_m"),
        "depth_mae_m": ((evaluation.get("depth") or {}).get("ground_truth_water_region") or {}).get("mae_m"),
        "visual_gate_status": evaluation["gate"]["visual_gate_status"],
        "geometry_gate_status": evaluation["gate"]["geometry_gate_status"],
        "geometry_reject_reasons": list(geometry_gate.get("reasons", [])),
        "visual_reject_or_partial_reasons": list(visual_gate.get("reasons", [])),
        "unknown_fraction": visual_gate.get("unknown_fraction"),
        "water_mask_time_stability": (visual_gate.get("metrics") or {}).get("water_mask_time_stability"),
        "trusted_shoreline_fraction": ray_diagnostics.get("trusted_shoreline_fraction"),
        "shoreline_intersection_success_rate": ray_diagnostics.get("shoreline_intersection_success_rate"),
        "ray_dem_intersection_count": len(shoreline_document.get("intersections", [])),
        "ray_dem_absolute_residual_p95_m": _percentile(absolute_ray_residuals, 95),
        "ray_dem_absolute_residual_max_m": max(absolute_ray_residuals, default=None),
        **boundary,
        **phase,
        "threshold_px": BOUNDARY_GATE_THRESHOLD_PX,
        "threshold_exceeded": bool(recomputed_p95 is not None and recomputed_p95 > BOUNDARY_GATE_THRESHOLD_PX),
        "prediction_rerun": False,
        "ground_truth_modified": False,
        "eligible_for_downstream": False,
    }


def _metric_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    reportable_phase = [row for row in records if row["phase_alignment_reporting_level_met"]]
    return {
        "sequence_count": len(records),
        "boundary_p95_min_px": min(_finite(row["boundary_reprojection_p95_px"] for row in records), default=None),
        "boundary_p95_median_px": _median(row["boundary_reprojection_p95_px"] for row in records),
        "boundary_p95_mean_px": _mean(row["boundary_reprojection_p95_px"] for row in records),
        "boundary_p95_max_px": max(_finite(row["boundary_reprojection_p95_px"] for row in records), default=None),
        "camera_mask_iou_mean": _mean(row["camera_mask_iou"] for row in records),
        "camera_reprojection_iou_mean": _mean(row["camera_reprojection_iou"] for row in records),
        "water_level_absolute_error_m_mean": _mean(row["water_level_absolute_error_m"] for row in records),
        "depth_mae_m_mean": _mean(row["depth_mae_m"] for row in records),
        "unknown_fraction_mean": _mean(row["unknown_fraction"] for row in records),
        "water_mask_time_stability_mean": _mean(row["water_mask_time_stability"] for row in records),
        "phase_alignment_reportable_count": len(reportable_phase),
        "phase_alignment_dx_median_px": _median(row["phase_alignment_dx_px"] for row in reportable_phase),
        "phase_alignment_dy_median_px": _median(row["phase_alignment_dy_px"] for row in reportable_phase),
        "phase_alignment_response_median": _median(row["phase_alignment_response"] for row in records),
    }


def build_geometry_diagnostics_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: (row["case_id"], row["rain_level"], row["seed"]))
    p95 = _finite(row["boundary_reprojection_p95_px"] for row in ordered)
    reasons = Counter(reason for row in ordered for reason in row["geometry_reject_reasons"])
    strong_phase = [row for row in ordered if row["phase_alignment_reporting_level_met"]]
    def sign_fraction(values: Iterable[Any], positive: bool) -> float | None:
        finite = _finite(values)
        if not finite:
            return None
        return float(sum(value > 0 if positive else value < 0 for value in finite) / len(finite))
    high_iou = [row for row in ordered if row["camera_mask_iou"] >= 0.8]
    domain_deltas = _finite(row["unknown_aware_minus_full_image_p95_px"] for row in ordered)
    full_p95_values = _finite(row["full_image_boundary_reprojection_p95_px"] for row in ordered)
    summary = {
        "data_role": "diagnostic_summary",
        "sequence_count": len(ordered),
        "prediction_rerun": False,
        "threshold_px": BOUNDARY_GATE_THRESHOLD_PX,
        "threshold_modified": False,
        "p95_distribution": {
            "minimum_px": min(p95) if p95 else None,
            "median_px": _median(p95),
            "mean_px": _mean(p95),
            "maximum_px": max(p95) if p95 else None,
            "above_3_through_4_count": sum(BOUNDARY_GATE_THRESHOLD_PX < value <= 4 for value in p95),
            "above_5_count": sum(value > 5 for value in p95),
            "above_8_count": sum(value > 8 for value in p95),
            "above_10_count": sum(value > 10 for value in p95),
        },
        "by_depth": {
            str(depth): _metric_group([row for row in ordered if row["nominal_depth_cm"] == depth])
            for depth in sorted({row["nominal_depth_cm"] for row in ordered})
        },
        "by_rain_level": {
            rain: _metric_group([row for row in ordered if row["rain_level"] == rain])
            for rain in sorted({row["rain_level"] for row in ordered})
        },
        "correlations": {
            "p95_vs_nominal_depth_pearson": _pearson(
                (row["boundary_reprojection_p95_px"] for row in ordered),
                (row["nominal_depth_cm"] for row in ordered),
            ),
            "p95_vs_camera_mask_iou_pearson": _pearson(
                (row["boundary_reprojection_p95_px"] for row in ordered),
                (row["camera_mask_iou"] for row in ordered),
            ),
            "p95_vs_camera_reprojection_iou_pearson": _pearson(
                (row["boundary_reprojection_p95_px"] for row in ordered),
                (row["camera_reprojection_iou"] for row in ordered),
            ),
            "p95_vs_water_level_error_pearson": _pearson(
                (row["boundary_reprojection_p95_px"] for row in ordered),
                (row["water_level_absolute_error_m"] for row in ordered),
            ),
            "p95_vs_unknown_fraction_pearson": _pearson(
                (row["boundary_reprojection_p95_px"] for row in ordered),
                (row["unknown_fraction"] for row in ordered),
            ),
            "p95_vs_water_mask_time_stability_pearson": _pearson(
                (row["boundary_reprojection_p95_px"] for row in ordered),
                (row["water_mask_time_stability"] for row in ordered),
            ),
        },
        "unknown_aware_domain_comparison": {
            "comparison_role": "diagnostic_counterfactual_only_formal_gate_domain_unchanged",
            "unknown_aware_p95_median_px": _median(p95),
            "full_image_p95_median_px": _median(full_p95_values),
            "unknown_aware_minus_full_p95_mean_px": _mean(domain_deltas),
            "unknown_aware_minus_full_p95_median_px": _median(domain_deltas),
            "unknown_aware_p95_greater_count": sum(value > 1e-9 for value in domain_deltas),
            "full_image_p95_greater_count": sum(value < -1e-9 for value in domain_deltas),
            "equal_within_tolerance_count": sum(abs(value) <= 1e-9 for value in domain_deltas),
            "full_image_above_3_count": sum(value > BOUNDARY_GATE_THRESHOLD_PX for value in full_p95_values),
            "formal_unknown_aware_above_3_count": sum(value > BOUNDARY_GATE_THRESHOLD_PX for value in p95),
        },
        "ray_dem_intersection_numerics": {
            "minimum_success_rate": min(
                _finite(row.get("shoreline_intersection_success_rate") for row in ordered), default=None,
            ),
            "maximum_absolute_residual_p95_m": max(
                _finite(row.get("ray_dem_absolute_residual_p95_m") for row in ordered), default=None,
            ),
            "maximum_absolute_residual_m": max(
                _finite(row.get("ray_dem_absolute_residual_max_m") for row in ordered), default=None,
            ),
            "interpretation_scope": "numerical_ray_dem_intersection_only_not_independent_camera_calibration_validation",
        },
        "global_translation_exploration": {
            "method": "phase_correlation_of_known_region_water_masks",
            "semantics": "exploratory_global_translation_not_boundary_correspondence",
            "reporting_response_level": PHASE_RESPONSE_REPORTING_LEVEL,
            "all_sequence_dx_median_px": _median(row["phase_alignment_dx_px"] for row in ordered),
            "all_sequence_dy_median_px": _median(row["phase_alignment_dy_px"] for row in ordered),
            "all_sequence_response_median": _median(row["phase_alignment_response"] for row in ordered),
            "response_level_met_count": len(strong_phase),
            "response_level_met_dx_median_px": _median(row["phase_alignment_dx_px"] for row in strong_phase),
            "response_level_met_dy_median_px": _median(row["phase_alignment_dy_px"] for row in strong_phase),
            "response_level_met_dx_positive_fraction": sign_fraction(
                (row["phase_alignment_dx_px"] for row in strong_phase), True,
            ),
            "response_level_met_dx_negative_fraction": sign_fraction(
                (row["phase_alignment_dx_px"] for row in strong_phase), False,
            ),
            "response_level_met_dy_positive_fraction": sign_fraction(
                (row["phase_alignment_dy_px"] for row in strong_phase), True,
            ),
            "response_level_met_dy_negative_fraction": sign_fraction(
                (row["phase_alignment_dy_px"] for row in strong_phase), False,
            ),
            "signed_nearest_neighbor_offsets_available": False,
            "signed_nearest_neighbor_offsets_reason": "nearest_boundary_matches_are_many_to_one_not_physical_correspondence",
        },
        "high_camera_iou_geometry_reject_count": len(high_iou),
        "high_camera_iou_geometry_reject_sequences": high_iou,
        "geometry_reject_reason_counts": dict(reasons.most_common()),
        "stored_recomputed_p95_max_delta_px": max(
            _finite(row["stored_recomputed_p95_absolute_delta_px"] for row in ordered), default=None,
        ),
        "eligible_for_downstream": False,
    }
    return summary


CSV_FIELDS = [
    "case_id", "nominal_depth_cm", "rain_level", "seed", "camera_mask_iou",
    "boundary_reprojection_p50_px", "boundary_reprojection_p95_px", "camera_reprojection_iou",
    "water_level_absolute_error_m", "depth_mae_m", "visual_gate_status", "geometry_reject_reasons",
    "signed_dx_mean_px", "signed_dy_mean_px", "signed_dx_median_px", "signed_dy_median_px",
    "phase_alignment_dx_px", "phase_alignment_dy_px", "phase_alignment_response",
    "unknown_fraction", "water_mask_time_stability", "stored_recomputed_p95_absolute_delta_px",
    "full_image_boundary_reprojection_p95_px", "unknown_aware_minus_full_image_p95_px",
    "shoreline_intersection_success_rate", "ray_dem_absolute_residual_p95_m", "ray_dem_absolute_residual_max_m",
]


def write_geometry_diagnostics(output_root: str | Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda row: (row["case_id"], row["rain_level"], row["seed"]))
    summary = build_geometry_diagnostics_summary(ordered)
    (output / "geometry_diagnostics.json").write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    with (output / "geometry_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in ordered:
            row = {field: record.get(field) for field in CSV_FIELDS}
            row["geometry_reject_reasons"] = ";".join(record["geometry_reject_reasons"])
            writer.writerow(row)
    write_figures_compatible(output)
    return summary


def write_figures_compatible(output: Path) -> None:
    if int(np.__version__.split(".", 1)[0]) >= 2 and os.environ.get("PHASE2DB1_MPL_COMPAT") != "1":
        environment = dict(os.environ)
        environment.update({"PYTHONNOUSERSITE": "1", "PHASE2DB1_MPL_COMPAT": "1"})
        subprocess.run(
            ["/usr/bin/python3", "-m", "src.evaluation.analyze_geometry_reprojection_error",
             "--render-existing", str(output)],
            check=True, env=environment,
        )
        return
    write_figures(output)


def write_figures(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = json.loads((output / "geometry_diagnostics.json").read_text(encoding="utf-8"))

    def save(figure, filename: str) -> None:
        figure.tight_layout()
        figure.savefig(output / filename, dpi=150)
        plt.close(figure)

    depths = sorted({row["nominal_depth_cm"] for row in records})
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot([[row["boundary_reprojection_p95_px"] for row in records if row["nominal_depth_cm"] == depth] for depth in depths], labels=[f"{depth} cm" for depth in depths])
    axis.axhline(BOUNDARY_GATE_THRESHOLD_PX, color="red", linestyle="--", label="3 px gate")
    axis.set_ylabel("Boundary reprojection P95 (px)"); axis.set_title("Boundary P95 by nominal depth"); axis.legend()
    save(figure, "boundary_p95_by_depth.png")

    rains = sorted({row["rain_level"] for row in records})
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot([[row["boundary_reprojection_p95_px"] for row in records if row["rain_level"] == rain] for rain in rains], labels=rains)
    axis.axhline(BOUNDARY_GATE_THRESHOLD_PX, color="red", linestyle="--", label="3 px gate")
    axis.set_ylabel("Boundary reprojection P95 (px)"); axis.set_title("Boundary P95 by rain level"); axis.legend()
    save(figure, "boundary_p95_by_rain_level.png")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for rain in rains:
        group = [row for row in records if row["rain_level"] == rain]
        axis.scatter([row["camera_mask_iou"] for row in group], [row["boundary_reprojection_p95_px"] for row in group], label=rain)
    axis.axhline(BOUNDARY_GATE_THRESHOLD_PX, color="red", linestyle="--")
    axis.set_xlabel("Camera mask IoU"); axis.set_ylabel("Boundary P95 (px)"); axis.set_title("Camera IoU vs boundary P95"); axis.legend()
    save(figure, "camera_iou_vs_boundary_p95.png")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.scatter([row["water_level_absolute_error_m"] for row in records], [row["boundary_reprojection_p95_px"] for row in records])
    axis.axhline(BOUNDARY_GATE_THRESHOLD_PX, color="red", linestyle="--")
    axis.set_xlabel("Water-level absolute error (m)"); axis.set_ylabel("Boundary P95 (px)"); axis.set_title("Water-level error vs boundary P95")
    save(figure, "water_level_error_vs_boundary_p95.png")

    reasons = sorted({reason for row in records for reason in row["geometry_reject_reasons"]})
    matrix = np.asarray([[sum(reason in row["geometry_reject_reasons"] for row in records if row["rain_level"] == rain) for rain in rains] for reason in reasons], dtype=int)
    figure, axis = plt.subplots(figsize=(8, max(4.5, 0.45 * len(reasons))))
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(rains)), rains); axis.set_yticks(range(len(reasons)), reasons)
    axis.set_title("Geometry reject reason matrix by rain level")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Sequence count")
    save(figure, "geometry_reject_reason_matrix.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-existing")
    args = parser.parse_args()
    if not args.render_existing:
        parser.error("--render-existing is required when invoking this module directly")
    write_figures(Path(args.render_existing).expanduser().resolve())


if __name__ == "__main__":
    main()
