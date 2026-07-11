from src.perception.temporal_event_label_matching import match_tracks_to_events


CONFIG = {
    "max_center_distance_px": 25, "minimum_temporal_iou": .05,
    "minimum_lifetime_overlap": .2, "minimum_spatial_score": .05,
    "ambiguity_margin": .05,
}


def _track(track_id, center, start=2, end=8):
    return {"track_id": track_id, "center_mean": center, "start_frame": start,
            "end_frame": end, "bbox_union": [center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4]}


def test_robust_matching_is_one_to_one_and_noise_is_explicit():
    tracks = [_track("a", [10, 10]), _track("b", [80, 80])]
    events = [{"event_id": "e", "event_type": "water_ripple", "center_u": 11,
               "center_v": 10, "start_frame": 1, "end_frame": 9, "radius_px": 5}]
    labels, diagnostics = match_tracks_to_events(tracks, events, CONFIG)
    assert labels[0]["label"] == "water_ripple"
    assert labels[1]["label"] == "background_noise"
    assert diagnostics["matched_track_count"] == 1
    assert diagnostics["noise_track_count"] == 1
