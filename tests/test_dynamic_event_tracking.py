from src.perception.dynamic_event_tracking import track_dynamic_events


def _candidate(frame, x):
    return {
        "candidate_id": f"c{frame}", "frame_index": frame, "center_u": float(x),
        "center_v": 10.0, "bbox": [x - 1, 9, x + 1, 11], "area": 9,
        "equivalent_radius": 1.7, "mean_residual": 12.0, "polarity": "positive",
        "ringness": 1.0, "compactness": 0.8,
    }


def test_nearby_candidates_form_deterministic_track():
    candidates = [[_candidate(0, 10)], [_candidate(1, 12)], [_candidate(2, 14)]]
    config = {"max_gap_frames": 1, "max_center_distance_px": 5, "min_bbox_iou_for_distant_match": 0.01}
    first = track_dynamic_events(candidates, config)
    second = track_dynamic_events(candidates, config)
    assert first == second
    assert len(first) == 1
    assert first[0]["duration_frames"] == 3
