#!/usr/bin/env python3
"""Deterministic summaries and standalone figures for Phase 2D-A evaluation."""

from __future__ import annotations

import csv
import os
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _mean(values) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return None if not finite else float(np.mean(finite))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def flatten_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    gate = evaluation["gate"]
    camera = evaluation["camera_mask"]
    depth = evaluation.get("depth") or {}
    return {
        "case_id": evaluation["case_id"], "nominal_depth_cm": evaluation["nominal_depth_cm"],
        "rain_level": evaluation["rain_level"], "seed": evaluation["seed"],
        "visual_gate": gate["visual_gate_status"], "geometry_gate": gate["geometry_gate_status"],
        "integration_gate": gate["integration_gate_status"], "measurement_status": gate["measurement_status"],
        "authoritative_measurement_available": gate["authoritative_measurement_available"],
        "camera_mask_iou": camera["whole_image_iou"], "known_region_iou": camera["known_region_iou"],
        "water_level_absolute_error_m": (evaluation.get("water_level") or {}).get("water_level_absolute_error_m"),
        "dem_mask_iou": (evaluation.get("dem_mask") or {}).get("iou"),
        "depth_mae_m": (depth.get("ground_truth_water_region") or {}).get("mae_m"),
        "area_relative_error": (evaluation.get("area") or {}).get("relative_error"),
        "volume_relative_error": (evaluation.get("volume") or {}).get("relative_error"),
        "reject_reasons": ";".join(gate["reject_reasons"]),
        "metric_role": evaluation["metric_role"],
    }


def _metric_group(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sequence_count": len(evaluations),
        "camera_mask_iou_mean": _mean(item["camera_mask"]["whole_image_iou"] for item in evaluations),
        "known_region_iou_mean": _mean(item["camera_mask"]["known_region_iou"] for item in evaluations),
        "water_level_absolute_error_m_mean": _mean((item.get("water_level") or {}).get("water_level_absolute_error_m") for item in evaluations),
        "dem_mask_iou_mean": _mean((item.get("dem_mask") or {}).get("iou") for item in evaluations),
        "depth_mae_m_mean": _mean(((item.get("depth") or {}).get("ground_truth_water_region") or {}).get("mae_m") for item in evaluations),
        "area_relative_error_mean": _mean((item.get("area") or {}).get("relative_error") for item in evaluations),
        "volume_relative_error_mean": _mean((item.get("volume") or {}).get("relative_error") for item in evaluations),
        "available_geometry_metric_count": sum(item.get("water_level") is not None for item in evaluations),
    }


def build_dataset_summary(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(evaluations, key=lambda item: (item["case_id"], item["rain_level"], item["seed"]))
    dry = [item for item in ordered if item["is_dry"]]
    water = [item for item in ordered if not item["is_dry"]]
    gate_distribution = {
        key: dict(Counter(item["gate"][key] for item in ordered))
        for key in ("visual_gate_status", "geometry_gate_status", "integration_gate_status")
    }
    measurement_distribution = dict(Counter(item["gate"]["measurement_status"] for item in ordered))
    reason_counts = Counter(reason for item in ordered for reason in item["gate"]["reject_reasons"])
    metrics_by_depth = {
        str(depth): _metric_group([item for item in water if item["nominal_depth_cm"] == depth])
        for depth in sorted({item["nominal_depth_cm"] for item in water})
    }
    metrics_by_rain = {
        rain: _metric_group([item for item in water if item["rain_level"] == rain])
        for rain in sorted({item["rain_level"] for item in water})
    }
    metrics_by_status = {
        status: _metric_group([item for item in ordered if item["gate"]["measurement_status"] == status])
        for status in sorted(measurement_distribution)
    }
    authoritative = sum(item["gate"]["authoritative_measurement_available"] for item in water)
    components_40 = [
        {"rain_level": item["rain_level"], "seed": item["seed"], "gate": item["gate"],
         "components": item["components_40cm"]}
        for item in water if item["case_id"] == "sim_water_40cm_001"
    ]
    visual_good_geometry_reject = [
        f"{item['case_id']}/{item['rain_level']}/seed_{item['seed']}"
        for item in water if item["camera_mask"]["whole_image_iou"] >= .8
        and item["gate"]["geometry_gate_status"] == "reject"
    ]
    visual_partial_geometry_available = [
        f"{item['case_id']}/{item['rain_level']}/seed_{item['seed']}"
        for item in water if item["gate"]["visual_gate_status"] == "partial"
        and item["gate"]["candidate_values_available"]
    ]
    summary = {
        "sequence_count": len(ordered), "dry_sequence_count": len(dry), "water_sequence_count": len(water),
        "gate_distribution": gate_distribution,
        "measurement_status_distribution": measurement_distribution,
        "authoritative_measurement_count": authoritative,
        "authoritative_measurement_success_rate_water_sequences": _mean([authoritative / len(water)]) if water else None,
        "dry_false_positive": {
            "total_false_water_pixels": sum(item["dry_false_positive"]["false_water_pixels"] for item in dry),
            "mean_false_water_fraction": _mean(item["dry_false_positive"]["false_water_fraction"] for item in dry),
            "total_false_water_components": sum(item["dry_false_positive"]["false_water_components"] for item in dry),
            "mean_false_positive_area_m2": _mean(item["dry_false_positive"]["false_positive_area_m2"] for item in dry),
        },
        "metrics_by_depth": metrics_by_depth, "metrics_by_rain_level": metrics_by_rain,
        "metrics_by_measurement_status": metrics_by_status,
        "failure_reasons": dict(reason_counts.most_common()),
        "most_common_reject_reason": reason_counts.most_common(1)[0] if reason_counts else None,
        "components_40cm": components_40,
        "visual_mask_good_but_geometry_reject_sequences": visual_good_geometry_reject,
        "visual_partial_with_candidate_geometry_sequences": visual_partial_geometry_available,
        "boundary_p95_threshold_evidence": {
            "reject_count": reason_counts.get("geometry:boundary_reprojection_error_above_threshold", 0),
            "threshold_changed": False,
        },
        "missing_values_policy": "null_and_excluded_from_means_never_replaced_by_zero",
        "eligible_for_downstream": False,
    }
    return summary


def write_summary_outputs(output_root: str | Path, evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_dataset_summary(evaluations)
    rows = [flatten_evaluation(item) for item in sorted(evaluations, key=lambda item: (item["case_id"], item["rain_level"], item["seed"]))]
    _write_json(output / "dataset_summary.json", summary)
    _write_json(output / "gate_distribution.json", summary["gate_distribution"])
    _write_json(output / "metrics_by_depth.json", summary["metrics_by_depth"])
    _write_json(output / "metrics_by_rain_level.json", summary["metrics_by_rain_level"])
    _write_json(output / "metrics_by_measurement_status.json", summary["metrics_by_measurement_status"])
    _write_json(output / "failure_reasons.json", summary["failure_reasons"])
    _write_json(output / "dataset_evaluations.json", evaluations)
    with (output / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    write_figures_compatible(output, evaluations, summary)
    return summary


def write_figures_compatible(output: Path, evaluations: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Use the system NumPy-1 runtime only when local Matplotlib is ABI-incompatible."""
    numpy_major = int(str(np.__version__).split(".", 1)[0])
    if numpy_major >= 2 and os.environ.get("PHASE2D_MPL_COMPAT") != "1":
        environment = dict(os.environ)
        environment.update({"PYTHONNOUSERSITE": "1", "PHASE2D_MPL_COMPAT": "1"})
        subprocess.run(
            ["/usr/bin/python3", "-m", "src.evaluation.synthetic_visual_to_depth_summary", "--render-existing", str(output)],
            check=True, env=environment,
        )
        return
    write_figures(output, evaluations, summary)


def write_figures(output: Path, evaluations: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_bar(name: str, labels: list[str], values: list[float], ylabel: str, title: str):
        figure = plt.figure(figsize=(7, 4.5))
        axis = figure.add_subplot(111)
        axis.bar(labels, values)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        figure.tight_layout()
        figure.savefig(output / name, dpi=150)
        plt.close(figure)

    depths = sorted(summary["metrics_by_depth"], key=int)
    for filename, key, ylabel, title in (
        ("camera_mask_iou_by_depth.png", "camera_mask_iou_mean", "Mean IoU", "Camera mask IoU by nominal depth"),
        ("water_level_error_by_depth.png", "water_level_absolute_error_m_mean", "Mean absolute error (m)", "Water-level error by nominal depth"),
        ("depth_mae_by_depth.png", "depth_mae_m_mean", "Mean MAE (m)", "Depth MAE by nominal depth"),
    ):
        pairs = [(depth, summary["metrics_by_depth"][depth][key]) for depth in depths]
        pairs = [(depth, value) for depth, value in pairs if value is not None]
        save_bar(filename, [f"{depth}cm" for depth, _ in pairs], [value for _, value in pairs], ylabel, title)
    integration_distribution = summary["gate_distribution"]["integration_gate_status"]
    save_bar("gate_status_distribution.png", list(integration_distribution), list(integration_distribution.values()), "Sequence count", "Integration gate status distribution")
    rains = sorted(summary["metrics_by_rain_level"])
    rain_pairs = [(rain, summary["metrics_by_rain_level"][rain]["camera_mask_iou_mean"]) for rain in rains]
    rain_pairs = [(rain, value) for rain, value in rain_pairs if value is not None]
    save_bar("metrics_by_rain_level.png", [rain for rain, _ in rain_pairs], [value for _, value in rain_pairs], "Mean Camera IoU", "Camera-mask metric by rain level")
    dry = [item for item in evaluations if item["is_dry"]]
    save_bar(
        "dry_false_positive_summary.png",
        [f"{item['rain_level']}-s{item['seed']}" for item in dry],
        [item["dry_false_positive"]["false_water_pixels"] for item in dry],
        "False water pixels", "Dry-sequence false positives",
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Render existing Phase 2D-A summaries with Matplotlib")
    parser.add_argument("--render-existing", required=True)
    args = parser.parse_args()
    output = Path(args.render_existing).expanduser().resolve()
    evaluations = json.loads((output / "dataset_evaluations.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "dataset_summary.json").read_text(encoding="utf-8"))
    write_figures(output, evaluations, summary)


if __name__ == "__main__":
    main()
