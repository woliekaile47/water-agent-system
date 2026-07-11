import numpy as np

from src.perception.synthetic_rain_visual_generator import generate_event_schedule
from tests.dynamic_rain_test_utils import base_and_mask, small_config


def test_event_centres_and_lifetimes_obey_water_semantics():
    _, mask = base_and_mask()
    events = generate_event_schedule(mask, small_config(), "moderate", 42)
    dry = [event for event in events if event["event_type"] == "dry_splash"]
    water = [event for event in events if event["event_type"] == "water_ripple"]
    assert dry and water
    assert all(not mask[event["center_v"], event["center_u"]] for event in dry)
    assert all(mask[event["center_v"], event["center_u"]] for event in water)
    dry_lifetimes = [event["end_frame"] - event["start_frame"] + 1 for event in dry]
    water_lifetimes = [event["end_frame"] - event["start_frame"] + 1 for event in water]
    assert np.mean(dry_lifetimes) < np.mean(water_lifetimes)
    assert all(event["expansion_rate_px_per_frame"] > 0 for event in water)
    assert all(0 < event["damping_factor"] < 1 for event in water)


def test_empty_and_full_water_masks_are_safe():
    config = small_config()
    empty = np.zeros((32, 48), dtype=bool)
    full = np.ones((32, 48), dtype=bool)
    dry_events = generate_event_schedule(empty, config, "moderate", 42)
    water_events = generate_event_schedule(full, config, "moderate", 42)
    assert all(event["event_type"] == "dry_splash" for event in dry_events)
    assert all(event["event_type"] == "water_ripple" for event in water_events)


def test_light_moderate_heavy_event_counts_increase():
    _, mask = base_and_mask()
    config = small_config()
    counts = [len(generate_event_schedule(mask, config, level, 42)) for level in ("light", "moderate", "heavy")]
    assert counts[0] < counts[1] < counts[2]
