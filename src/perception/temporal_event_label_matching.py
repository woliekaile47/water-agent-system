#!/usr/bin/env python3
"""Training-only robust one-to-one matching between RGB tracks and GT events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_training_event_labels(sequence_dir: str | Path) -> list[dict[str, Any]]:
    """Load only event fields needed for offline training; never used by inference."""
    path = Path(sequence_dir).expanduser().resolve() / "ground_truth" / "event_annotations.json"
    with path.open("r", encoding="utf-8") as stream:
        events = (json.load(stream) or {}).get("events", [])
    allowed = ("event_id", "event_type", "center_u", "center_v", "start_frame", "end_frame", "radius_px")
    return [{key: event[key] for key in allowed if key in event} for event in events]


def _pair_metrics(track: dict, event: dict, config: dict) -> dict[str, float] | None:
    intersection = max(0, min(track["end_frame"], event["end_frame"]) - max(track["start_frame"], event["start_frame"]) + 1)
    if intersection <= 0:
        return None
    track_duration = track["end_frame"] - track["start_frame"] + 1
    event_duration = event["end_frame"] - event["start_frame"] + 1
    temporal_iou = intersection / max(1, track_duration + event_duration - intersection)
    event_overlap = intersection / max(1, event_duration)
    center_error = float(np.hypot(track["center_mean"][0] - event["center_u"], track["center_mean"][1] - event["center_v"]))
    max_distance = float(config["max_center_distance_px"])
    x0, y0, x1, y1 = track["bbox_union"]
    closest_x = min(max(float(event["center_u"]), x0), x1)
    closest_y = min(max(float(event["center_v"]), y0), y1)
    bbox_distance = float(np.hypot(closest_x - event["center_u"], closest_y - event["center_v"]))
    radius = max(1.0, float(event.get("radius_px", 1.0)))
    spatial_score = max(0.0, 1.0 - bbox_distance / (radius + max_distance * 0.5))
    distance_score = max(0.0, 1.0 - center_error / max_distance)
    if center_error > max_distance:
        return None
    if temporal_iou < float(config["minimum_temporal_iou"]) and event_overlap < float(config["minimum_lifetime_overlap"]):
        return None
    if spatial_score < float(config["minimum_spatial_score"]):
        return None
    score = 0.30 * temporal_iou + 0.25 * event_overlap + 0.30 * distance_score + 0.15 * spatial_score
    return {
        "score": float(score), "center_error": center_error,
        "temporal_iou": float(temporal_iou), "event_overlap": float(event_overlap),
        "spatial_score": float(spatial_score),
    }


def match_tracks_to_events(
    tracks: list[dict[str, Any]], events: list[dict[str, Any]], config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    per_track_scores: dict[int, list[float]] = {}
    for track_index, track in enumerate(tracks):
        for event_index, event in enumerate(events):
            metrics = _pair_metrics(track, event, config)
            if metrics is None:
                continue
            candidates.append({"track_index": track_index, "event_index": event_index, **metrics})
            per_track_scores.setdefault(track_index, []).append(metrics["score"])
    ambiguity_margin = float(config["ambiguity_margin"])
    ambiguous_tracks = {
        index for index, scores in per_track_scores.items()
        if len(scores) > 1 and sorted(scores, reverse=True)[0] - sorted(scores, reverse=True)[1] < ambiguity_margin
    }
    assigned_tracks: set[int] = set()
    assigned_events: set[int] = set()
    assignments: dict[int, dict[str, Any]] = {}
    for candidate in sorted(candidates, key=lambda item: (-item["score"], item["track_index"], item["event_index"])):
        track_index, event_index = candidate["track_index"], candidate["event_index"]
        if track_index in ambiguous_tracks or track_index in assigned_tracks or event_index in assigned_events:
            continue
        assigned_tracks.add(track_index)
        assigned_events.add(event_index)
        event = events[event_index]
        assignments[track_index] = {
            "label": event["event_type"], "event_id": event.get("event_id"),
            "match_score": candidate["score"], "center_error": candidate["center_error"],
            "temporal_iou": candidate["temporal_iou"], "event_overlap": candidate["event_overlap"],
        }
    labeled = []
    for index, track in enumerate(tracks):
        if index in assignments:
            label = assignments[index]
        elif index in ambiguous_tracks:
            label = {"label": "uncertain", "event_id": None, "match_score": 0.0}
        else:
            label = {"label": "background_noise", "event_id": None, "match_score": 0.0}
        labeled.append({"track_id": track["track_id"], **label})
    matched = list(assignments.values())
    diagnostics = {
        "matched_track_count": len(matched),
        "unmatched_track_count": len(tracks) - len(matched) - len(ambiguous_tracks),
        "ambiguous_match_count": len(ambiguous_tracks),
        "dry_match_count": sum(item["label"] == "dry_splash" for item in matched),
        "water_match_count": sum(item["label"] == "water_ripple" for item in matched),
        "noise_track_count": sum(item["label"] == "background_noise" for item in labeled),
        "mean_center_error": float(np.mean([item["center_error"] for item in matched])) if matched else None,
        "mean_temporal_overlap": float(np.mean([item["event_overlap"] for item in matched])) if matched else None,
        "unmatched_event_count": len(events) - len(assigned_events),
    }
    return labeled, diagnostics
