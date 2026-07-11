#!/usr/bin/env python3
"""Deterministic greedy spatiotemporal tracking of dynamic candidates."""

from __future__ import annotations

from typing import Any

import numpy as np


def _bbox_iou(left: list[int], right: list[int]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)
    left_area = max(0, left[2] - left[0] + 1) * max(0, left[3] - left[1] + 1)
    right_area = max(0, right[2] - right[0] + 1) * max(0, right[3] - right[1] + 1)
    return float(intersection / max(1, left_area + right_area - intersection))


def track_dynamic_events(
    candidates_by_frame: list[list[dict[str, Any]]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    max_gap = int(config["max_gap_frames"])
    max_distance = float(config["max_center_distance_px"])
    for frame_index, candidates in enumerate(candidates_by_frame):
        available_tracks = [
            index for index, track in enumerate(tracks)
            if frame_index - int(track["observations"][-1]["frame_index"]) <= max_gap + 1
        ]
        assigned: set[int] = set()
        for candidate in sorted(candidates, key=lambda item: (-item["area"], item["candidate_id"])):
            best_index = None
            best_cost = float("inf")
            for track_index in available_tracks:
                if track_index in assigned:
                    continue
                previous = tracks[track_index]["observations"][-1]
                distance = float(np.hypot(candidate["center_u"] - previous["center_u"], candidate["center_v"] - previous["center_v"]))
                overlap = _bbox_iou(candidate["bbox"], previous["bbox"])
                radius_allowance = 0.5 * (candidate["equivalent_radius"] + previous["equivalent_radius"])
                if distance > max_distance + radius_allowance and overlap < float(config["min_bbox_iou_for_distant_match"]):
                    continue
                area_ratio = abs(np.log(max(candidate["area"], 1) / max(previous["area"], 1)))
                cost = distance - overlap * 8.0 + area_ratio * 1.5
                if cost < best_cost:
                    best_cost, best_index = cost, track_index
            if best_index is None:
                tracks.append({"track_id": f"track_{len(tracks):05d}", "observations": [candidate]})
            else:
                tracks[best_index]["observations"].append(candidate)
                assigned.add(best_index)
    results = []
    for track in tracks:
        observations = track["observations"]
        centers = np.asarray([[item["center_u"], item["center_v"]] for item in observations])
        union = [
            min(item["bbox"][0] for item in observations), min(item["bbox"][1] for item in observations),
            max(item["bbox"][2] for item in observations), max(item["bbox"][3] for item in observations),
        ]
        result = {
            "track_id": track["track_id"],
            "start_frame": int(observations[0]["frame_index"]),
            "end_frame": int(observations[-1]["frame_index"]),
            "duration_frames": int(observations[-1]["frame_index"] - observations[0]["frame_index"] + 1),
            "center_mean": [float(np.mean(centers[:, 0])), float(np.mean(centers[:, 1]))],
            "center_drift": float(np.max(np.linalg.norm(centers - np.mean(centers, axis=0), axis=1))),
            "area_sequence": [int(item["area"]) for item in observations],
            "equivalent_radius_sequence": [float(item["equivalent_radius"]) for item in observations],
            "intensity_sequence": [float(item["mean_residual"]) for item in observations],
            "polarity_sequence": [item["polarity"] for item in observations],
            "bbox_union": union,
            "valid_observation_count": len(observations),
            "observations": observations,
        }
        results.append(result)
    return results
