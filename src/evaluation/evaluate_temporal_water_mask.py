#!/usr/bin/env python3
"""Independent post-prediction evaluation for temporal water detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def load_temporal_evaluation_ground_truth(sequence_dir: str | Path) -> dict[str, Any]:
    root = Path(sequence_dir).expanduser().resolve()
    gt = root / "ground_truth"
    with (gt / "event_annotations.json").open("r", encoding="utf-8") as stream:
        annotations = json.load(stream)
    return {
        "water_mask": np.asarray(Image.open(gt / "water_mask.png").convert("L"), dtype=np.uint8) > 127,
        "event_map_sequence": np.load(gt / "event_map_sequence.npy"),
        "events": annotations["events"],
        "paths": {
            "water_mask": str(gt / "water_mask.png"),
            "event_map_sequence": str(gt / "event_map_sequence.npy"),
            "event_annotations": str(gt / "event_annotations.json"),
        },
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_water_mask(predicted: np.ndarray, unknown: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(predicted, dtype=bool)
    unknown_mask = np.asarray(unknown, dtype=bool)
    gt = np.asarray(truth, dtype=bool)
    tp = int(np.count_nonzero(prediction & gt))
    fp = int(np.count_nonzero(prediction & ~gt))
    fn = int(np.count_nonzero(~prediction & gt))
    union = int(np.count_nonzero(prediction | gt))
    kernel = np.ones((3, 3), np.uint8)
    pred_boundary = prediction & ~cv2.erode(prediction.astype(np.uint8), kernel).astype(bool)
    gt_boundary = gt & ~cv2.erode(gt.astype(np.uint8), kernel).astype(bool)
    pred_dilated = cv2.dilate(pred_boundary.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    boundary_precision = _ratio(np.count_nonzero(pred_boundary & gt_dilated), np.count_nonzero(pred_boundary))
    boundary_recall = _ratio(np.count_nonzero(gt_boundary & pred_dilated), np.count_nonzero(gt_boundary))
    known = ~unknown_mask
    known_union = int(np.count_nonzero((prediction | gt) & known))
    known_intersection = int(np.count_nonzero(prediction & gt & known))
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": _ratio(2 * precision * recall, precision + recall),
        "whole_image_iou": _ratio(tp, union),
        "boundary_f1": _ratio(2 * boundary_precision * boundary_recall, boundary_precision + boundary_recall),
        "predicted_water_area_pixels": int(np.count_nonzero(prediction)),
        "gt_water_area_pixels": int(np.count_nonzero(gt)),
        "unknown_fraction": float(np.mean(unknown_mask)),
        "evaluated_known_region_iou": _ratio(known_intersection, known_union),
        "observable_water_region_iou": _ratio(tp, union),
        "metric_domains": {
            "whole_image": "all Camera pixels; unknown is not silently treated as correct dry",
            "known_evidence_region": "pixels not marked unknown",
            "observable_water_region": "Camera Ground Truth water region and prediction union",
        },
    }


def _temporal_overlap(track: dict, event: dict) -> int:
    return max(0, min(track["end_frame"], event["end_frame"]) - max(track["start_frame"], event["start_frame"]) + 1)


def evaluate_events(classifications: list[dict], truth_events: list[dict], max_center_distance: float = 20.0) -> dict[str, Any]:
    predictions = [item for item in classifications if item["classification"] != "uncertain"]
    matches = []
    used_truth: set[int] = set()
    for prediction in sorted(predictions, key=lambda item: -item["confidence"]):
        best = None
        best_cost = float("inf")
        for index, event in enumerate(truth_events):
            if index in used_truth or _temporal_overlap(prediction, event) <= 0:
                continue
            distance = float(np.hypot(prediction["center_mean"][0] - event["center_u"], prediction["center_mean"][1] - event["center_v"]))
            if distance > max_center_distance:
                continue
            cost = distance - _temporal_overlap(prediction, event) * 0.2
            if cost < best_cost:
                best, best_cost = index, cost
        if best is not None:
            used_truth.add(best)
            event = truth_events[best]
            matches.append((prediction, event, float(np.hypot(prediction["center_mean"][0] - event["center_u"], prediction["center_mean"][1] - event["center_v"]))))
    classes = ("dry_splash", "water_ripple")
    per_class = {}
    confusion = {truth: {prediction: 0 for prediction in classes} for truth in classes}
    for class_name in classes:
        predicted_count = sum(item["classification"] == class_name for item in predictions)
        gt_count = sum(item["event_type"] == class_name for item in truth_events)
        correct = sum(pred["classification"] == class_name and event["event_type"] == class_name for pred, event, _ in matches)
        precision, recall = _ratio(correct, predicted_count), _ratio(correct, gt_count)
        per_class[class_name] = {
            "precision": precision, "recall": recall,
            "f1": _ratio(2 * precision * recall, precision + recall),
            "predicted_count": predicted_count, "ground_truth_count": gt_count,
        }
    for prediction, event, _ in matches:
        if event["event_type"] in confusion and prediction["classification"] in confusion[event["event_type"]]:
            confusion[event["event_type"]][prediction["classification"]] += 1
    center_errors = [distance for _, _, distance in matches]
    lifetime_errors = [abs(pred["duration_frames"] - (event["end_frame"] - event["start_frame"] + 1)) for pred, event, _ in matches]
    water_gt = sum(item["event_type"] == "water_ripple" for item in truth_events)
    matched_water_gt = sum(event["event_type"] == "water_ripple" for _, event, _ in matches)
    return {
        "per_class": per_class,
        "confusion_matrix": confusion,
        "matched_track_count": len(matches),
        "mean_center_localization_error_px": float(np.mean(center_errors)) if center_errors else None,
        "mean_lifetime_error_frames": float(np.mean(lifetime_errors)) if lifetime_errors else None,
        "water_event_coverage": _ratio(matched_water_gt, water_gt),
        "false_water_events_on_dry_sequence": sum(item["classification"] == "water_ripple" for item in predictions) if water_gt == 0 else 0,
    }


def evaluate_labeled_track_classifications(
    classifications: list[dict], track_labels: list[dict],
) -> dict[str, Any]:
    """Evaluate classifications after an independent robust track/event matching step."""
    label_by_track = {item["track_id"]: item["label"] for item in track_labels}
    classes = ("dry_splash", "water_ripple")
    result = {}
    confusion = {
        truth: {prediction: 0 for prediction in ("dry_splash", "water_ripple", "uncertain")}
        for truth in ("dry_splash", "water_ripple", "background_noise")
    }
    for item in classifications:
        truth = label_by_track.get(item["track_id"], "background_noise")
        if truth == "uncertain":
            continue
        confusion[truth][item["classification"]] += 1
    for class_name in classes:
        tp = sum(
            label_by_track.get(item["track_id"]) == class_name and item["classification"] == class_name
            for item in classifications
        )
        fp = sum(
            label_by_track.get(item["track_id"]) != class_name and item["classification"] == class_name
            for item in classifications
        )
        fn = sum(
            label_by_track.get(item["track_id"]) == class_name and item["classification"] != class_name
            for item in classifications
        )
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        result[class_name] = {
            "precision": precision, "recall": recall,
            "f1": _ratio(2 * precision * recall, precision + recall),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        }
    return {
        "per_class": result,
        "macro_f1": float(np.mean([result[name]["f1"] for name in classes])),
        "confusion_matrix": confusion,
        "uncertain_rate": _ratio(sum(item["classification"] == "uncertain" for item in classifications), len(classifications)),
        "background_noise_false_water": sum(
            label_by_track.get(item["track_id"], "background_noise") == "background_noise"
            and item["classification"] == "water_ripple" for item in classifications
        ),
    }
