from pathlib import Path

import numpy as np


def small_config():
    levels = {}
    for name, rate, seed in (("light", 1.0, 1), ("moderate", 2.0, 2), ("heavy", 3.0, 3)):
        levels[name] = {
            "random_seed": seed, "event_rate_hz": rate,
            "dry_event_fraction": 0.5, "water_event_fraction": 0.5,
            "dry_radius_px": [2, 3], "dry_lifetime_frames": [2, 3],
            "water_initial_radius_px": [2, 3], "water_expansion_rate_px_per_frame": [1.0, 1.5],
            "water_lifetime_frames": [4, 6], "brightness_disturbance": 0.02,
            "gaussian_noise_sigma": 0.5, "rain_streaks_per_frame": int(rate),
        }
    return {
        "schema_version": "1.0", "generator_version": "test_v1",
        "fps": 4, "duration_s": 2, "frame_count": 8,
        "deterministic_created_at": "2026-01-01T00:00:00+00:00",
        "rain_levels": levels,
        "event_model": {
            "center_margin_px": 2, "dry_spoke_count": [3, 4], "dry_satellite_count": [1, 2],
            "water_damping_factor": [0.85, 0.9], "water_ring_count": [1, 2],
            "water_ellipse_ratio": [0.8, 1.0], "water_intensity": [10.0, 15.0],
            "dry_intensity": [12.0, 18.0], "water_energy_outside_mask_fraction": 0.2,
        },
        "environment_perturbations": {
            "global_brightness": {"enabled": True, "amplitude": 0.01, "period_s": 2.0},
            "exposure_flicker": {"enabled": True, "amplitude": 0.005, "period_frames": 4},
            "gaussian_noise": {"enabled": True},
            "rain_streaks": {"enabled": True, "length_px": [2, 4], "intensity": 3.0},
            "water_reflection_flicker": {"enabled": True, "intensity": 2.0, "spatial_period_px": 10.0},
            "compression_artifact": {"enabled": True, "downscale_factor": 2, "blend": 0.05},
            "camera_jitter": {"enabled": False, "max_offset_px": 1},
        },
        "preview_stride_frames": 2,
    }


def base_and_mask():
    yy, xx = np.indices((32, 48))
    base = np.stack((80 + xx, 90 + yy, np.full_like(xx, 110)), axis=2).astype(np.uint8)
    mask = np.zeros((32, 48), dtype=bool)
    mask[10:25, 18:38] = True
    return base, mask


def source_metadata():
    return {"method": "unit_test", "depends_on_manual_screenshot": False}


def project_root():
    return Path(__file__).resolve().parents[1]
