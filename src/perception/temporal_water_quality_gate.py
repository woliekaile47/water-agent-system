#!/usr/bin/env python3
"""Prediction-side quality gate for sparse temporal water evidence."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def evaluate_temporal_quality_gate(
    loader: dict[str, Any],
    preprocessing: dict[str, Any],
    candidate_diagnostics: dict[str, Any],
    tracks: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    evidence: dict[str, np.ndarray],
    evidence_diagnostics: dict[str, Any],
    order_sensitivity: float,
    water_mask_time_stability: float,
    feature_score_separation: float,
    fps: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    hard_reasons: list[str] = []
    partial_reasons: list[str] = []
    frame_count = int(loader["frame_count"])
    high_water = sum(
        item["classification"] == "water_ripple"
        and item["confidence"] >= float(config["high_confidence_water_threshold"])
        for item in classifications
    )
    water_mask = np.asarray(evidence["predicted_water_mask"], dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        water_mask.astype(np.uint8), connectivity=8
    )
    component_areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)]
    largest_ratio = float(max(component_areas) / max(1, sum(component_areas))) if component_areas else 0.0
    if frame_count < int(config["min_input_frames"]):
        hard_reasons.append("insufficient_input_frames")
    if not (float(config["min_fps"]) <= fps <= float(config["max_fps"])):
        hard_reasons.append("fps_out_of_range")
    if not np.isfinite(evidence["predicted_water_probability"]).all():
        hard_reasons.append("nonfinite_probability")
    if len(tracks) < int(config["min_dynamic_tracks"]):
        partial_reasons.append("insufficient_dynamic_tracks")
    if high_water < int(config["min_high_confidence_water_tracks"]):
        partial_reasons.append("insufficient_high_confidence_water_tracks")
    if evidence_diagnostics["evidence_coverage_fraction"] < float(config["min_evidence_coverage"]):
        partial_reasons.append("water_evidence_coverage_too_low")
    if evidence_diagnostics["unknown_fraction"] > float(config["max_unknown_fraction_for_pass"]):
        partial_reasons.append("unknown_fraction_high")
    if candidate_diagnostics["candidate_pixel_fraction"] > float(config["max_candidate_pixel_fraction"]):
        partial_reasons.append("dynamic_residual_saturation")
    if len(component_areas) > int(config["max_water_components"]):
        partial_reasons.append("too_many_water_components")
    if component_areas and largest_ratio < float(config["min_largest_component_ratio"]):
        partial_reasons.append("largest_component_ratio_low")
    if evidence_diagnostics["evidence_concentration"] > float(config["max_evidence_concentration"]):
        partial_reasons.append("evidence_overly_concentrated")
    if preprocessing["exposure_anomaly_fraction"] > float(config["max_exposure_anomaly_fraction"]):
        partial_reasons.append("whole_frame_exposure_anomaly")
    if order_sensitivity < float(config["min_order_sensitivity"]):
        partial_reasons.append("temporal_order_sensitivity_too_low")
    if water_mask_time_stability < float(config["min_water_mask_time_stability"]):
        partial_reasons.append("water_mask_time_stability_low")
    if feature_score_separation < float(config["min_feature_score_separation"]):
        partial_reasons.append("temporal_features_not_separable")
    if evidence_diagnostics["predicted_water_fraction"] > float(config["max_predicted_water_fraction"]):
        partial_reasons.append("predicted_water_area_abnormally_large")
    if not np.any(water_mask):
        partial_reasons.append("predicted_water_mask_empty")
    if hard_reasons:
        status = "reject"
    elif partial_reasons:
        status = "partial"
    else:
        status = "pass"
    reasons = hard_reasons + partial_reasons
    temporal_sufficient = not any(
        reason in reasons for reason in (
            "insufficient_input_frames", "insufficient_dynamic_tracks",
            "insufficient_high_confidence_water_tracks", "water_evidence_coverage_too_low",
        )
    )
    return {
        "status": status,
        "reasons": reasons,
        "observable_region_result_valid": not hard_reasons,
        "temporal_evidence_sufficient": temporal_sufficient,
        "unknown_fraction": evidence_diagnostics["unknown_fraction"],
        "metrics": {
            "input_frame_count": frame_count,
            "fps": fps,
            "dynamic_track_count": len(tracks),
            "high_confidence_water_track_count": int(high_water),
            "water_track_count": evidence_diagnostics["water_track_count"],
            "evidence_coverage_fraction": evidence_diagnostics["evidence_coverage_fraction"],
            "predicted_water_fraction": evidence_diagnostics["predicted_water_fraction"],
            "water_component_count": len(component_areas),
            "largest_water_component_ratio": largest_ratio,
            "candidate_pixel_fraction": candidate_diagnostics["candidate_pixel_fraction"],
            "exposure_anomaly_fraction": preprocessing["exposure_anomaly_fraction"],
            "order_sensitivity": float(order_sensitivity),
            "water_mask_time_stability": float(water_mask_time_stability),
            "feature_score_separation": float(feature_score_separation),
        },
        "observation_source": "temporal_rain_impact_visual_evidence",
        "result_semantics": "camera_water_mask_from_sparse_temporal_evidence",
        "coverage_status": "insufficient" if hard_reasons else "partial" if evidence_diagnostics["unknown_fraction"] > 0.05 else "complete",
        "unknown_region_semantics": "no_temporal_evidence_not_confirmed_dry",
        "synthetic_domain": True,
        "real_world_validated": False,
        "eligible_for_downstream": False,
        "ground_truth_used": False,
    }


def evaluate_model_classifier_quality_gate(
    prediction: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    """Prediction-only model gate; probabilities and train-z values come from RGB tracks."""
    classifications = prediction["model_classifications"]
    baseline = prediction["baseline"]
    evidence = prediction["model_evidence"]
    diagnostics = prediction["model_evidence_diagnostics"]
    count = max(1, len(classifications))
    probabilities = np.asarray([item["model_water_probability"] for item in classifications], dtype=np.float64)
    uncertain_fraction = sum(item["classification"] == "uncertain" for item in classifications) / count
    confident_fraction = sum(item["confidence"] >= 0.65 for item in classifications) / count
    ood_fraction = sum(item["maximum_absolute_train_z"] > float(config["ood_z_threshold"]) for item in classifications) / count
    feature_values = np.asarray([
        [item.get(key, 0.0) for key in (
            "duration_frames", "maximum_area", "radius_growth_slope", "ringness",
            "temporal_energy", "local_dynamic_density", "neighbor_track_count",
        )] for item in classifications
    ], dtype=np.float64)
    nonfinite_fraction = float(np.mean(~np.isfinite(feature_values))) if feature_values.size else 0.0
    disagreement = sum(
        item["classification"] != item["baseline_rule_classification"]
        and item["classification"] != "uncertain"
        for item in classifications
    ) / count
    separation = float(np.mean(np.abs(probabilities - 0.5) * 2.0)) if probabilities.size else 0.0
    water_centers = np.asarray([
        item["center_mean"] for item in classifications if item["classification"] == "water_ripple"
    ], dtype=np.float64)
    cluster_spread = float(np.mean(np.std(water_centers, axis=0))) if water_centers.shape[0] > 1 else 0.0
    hard_reasons: list[str] = []
    partial_reasons: list[str] = []
    if nonfinite_fraction > float(config["max_nonfinite_feature_fraction"]):
        hard_reasons.append("nonfinite_track_features")
    if not np.isfinite(evidence["predicted_water_probability"]).all():
        hard_reasons.append("nonfinite_model_probability_map")
    if diagnostics["predicted_water_fraction"] > float(config["max_predicted_water_fraction"]):
        hard_reasons.append("model_water_area_abnormally_large")
    if uncertain_fraction > float(config["max_uncertain_fraction_for_pass"]):
        partial_reasons.append("classifier_uncertain_fraction_high")
    if confident_fraction < float(config["min_confident_track_fraction"]):
        partial_reasons.append("classifier_confidence_insufficient")
    if ood_fraction > float(config["max_ood_track_fraction"]):
        partial_reasons.append("track_feature_distribution_shift")
    if baseline["candidate_diagnostics"]["candidate_pixel_fraction"] > float(config["max_dynamic_candidate_pixel_fraction"]):
        partial_reasons.append("heavy_rain_dynamic_saturation")
    if diagnostics["maximum_single_track_contribution_fraction"] > float(config["max_single_track_contribution"]):
        partial_reasons.append("single_track_evidence_dominates")
    if prediction["model_window_mask_iou"] < float(config["min_window_mask_iou"]):
        partial_reasons.append("temporal_window_prediction_inconsistent")
    if disagreement > float(config["max_rule_model_disagreement"]):
        partial_reasons.append("rule_model_disagreement_high")
    if separation < float(config["min_probability_separation"]):
        partial_reasons.append("model_probability_separation_low")
    if not np.any(evidence["predicted_water_mask"]):
        partial_reasons.append("model_water_mask_empty")
    status = "reject" if hard_reasons else "partial" if partial_reasons else "pass"
    return {
        "status": status, "reasons": hard_reasons + partial_reasons,
        "temporal_evidence_sufficient": not any(reason in partial_reasons for reason in (
            "temporal_window_prediction_inconsistent", "model_water_mask_empty", "single_track_evidence_dominates",
        )),
        "classifier_confidence_sufficient": "classifier_confidence_insufficient" not in partial_reasons,
        "out_of_distribution_warning": "track_feature_distribution_shift" in partial_reasons,
        "coverage_status": "insufficient" if hard_reasons else "partial" if partial_reasons else "complete",
        "metrics": {
            "track_count": len(classifications), "uncertain_track_fraction": uncertain_fraction,
            "confident_track_fraction": confident_fraction, "ood_track_fraction": ood_fraction,
            "feature_anomaly_fraction": nonfinite_fraction, "water_track_cluster_spread_px": cluster_spread,
            "rule_model_disagreement_fraction": disagreement, "model_probability_separation": separation,
            "probability_mean": float(np.mean(probabilities)) if probabilities.size else 0.0,
            "probability_std": float(np.std(probabilities)) if probabilities.size else 0.0,
            "probability_calibration_proxy": "confidence_distribution_only_no_gt",
            "maximum_single_track_contribution_fraction": diagnostics["maximum_single_track_contribution_fraction"],
            "window_mask_iou": prediction["model_window_mask_iou"],
            "candidate_pixel_fraction": baseline["candidate_diagnostics"]["candidate_pixel_fraction"],
        },
        "observation_source": "temporal_rain_impact_visual_evidence",
        "ground_truth_used": False, "eligible_for_downstream": False,
    }
