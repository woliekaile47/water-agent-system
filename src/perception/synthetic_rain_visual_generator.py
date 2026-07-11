#!/usr/bin/env python3
"""Deterministic visual abstraction of rain impacts and water ripples.

Ground Truth is used only inside this synthetic sequence generator to place
and label events. Future detector code must consume only ``frames/``.
This is not a fluid-dynamics simulator or a validated real-rain model.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


GENERATOR_SOURCE = "deterministic_dynamic_rain_visual_abstraction"
_RESAMPLING = getattr(Image, "Resampling", Image)


def _decode_ros_image(message: Any) -> np.ndarray:
    height, width, step = int(message.height), int(message.width), int(message.step)
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(height, step)
    encoding = str(message.encoding).lower()
    if encoding in ("rgb8", "bgr8"):
        array = raw[:, : width * 3].reshape(height, width, 3).copy()
        return array[..., ::-1] if encoding == "bgr8" else array
    if encoding in ("rgba8", "bgra8"):
        array = raw[:, : width * 4].reshape(height, width, 4)[..., :3].copy()
        return array[..., ::-1] if encoding == "bgra8" else array
    if encoding == "mono8":
        mono = raw[:, :width].copy()
        return np.repeat(mono[..., None], 3, axis=2)
    raise ValueError(f"Unsupported ROS Image encoding: {message.encoding}")


def extract_stable_rgb_frame_from_rosbag(
    bag_dir: str | Path,
    topic: str,
    skip_camera_frames: int,
    median_frame_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a deterministic median Camera frame without replaying the bag."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image as RosImage

    bag_path = Path(bag_dir).expanduser().resolve()
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    frames: list[np.ndarray] = []
    timestamps: list[int] = []
    seen = 0
    while reader.has_next() and len(frames) < int(median_frame_count):
        current_topic, data, timestamp = reader.read_next()
        if current_topic != topic:
            continue
        if seen < int(skip_camera_frames):
            seen += 1
            continue
        message = deserialize_message(data, RosImage)
        frames.append(_decode_ros_image(message))
        timestamps.append(int(timestamp))
        seen += 1
    if len(frames) != int(median_frame_count):
        raise RuntimeError(
            f"Requested {median_frame_count} Camera frames after skip={skip_camera_frames}, got {len(frames)}"
        )
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f"Camera frames have inconsistent shapes: {sorted(shapes)}")
    stable = np.median(np.stack(frames).astype(np.float32), axis=0).round().astype(np.uint8)
    return stable, {
        "method": "rosbag2_read_only_pixel_median",
        "bag_dir": str(bag_path),
        "topic": topic,
        "skip_camera_frames": int(skip_camera_frames),
        "median_frame_count": int(median_frame_count),
        "source_timestamps_ns": timestamps,
        "encoding": "rgb8",
        "width": int(stable.shape[1]),
        "height": int(stable.shape[0]),
        "depends_on_manual_screenshot": False,
    }


def _eligible_centres(mask: np.ndarray, want_water: bool, margin: int) -> np.ndarray:
    target = np.asarray(mask, dtype=bool).copy() if want_water else ~np.asarray(mask, dtype=bool)
    if margin > 0:
        target[:margin, :] = False
        target[-margin:, :] = False
        target[:, :margin] = False
        target[:, -margin:] = False
    return np.column_stack(np.where(target))


def _integer_range(rng: np.random.Generator, values: list[int] | tuple[int, int]) -> int:
    low, high = int(values[0]), int(values[1])
    return int(rng.integers(low, high + 1))


def _float_range(rng: np.random.Generator, values: list[float] | tuple[float, float]) -> float:
    return float(rng.uniform(float(values[0]), float(values[1])))


def generate_event_schedule(
    water_mask: np.ndarray,
    config: dict[str, Any],
    rain_level: str,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Generate deterministic event annotations; no rendering side effects."""
    mask = np.asarray(water_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("water_mask must be a 2D array")
    level = config["rain_levels"][rain_level]
    model = config["event_model"]
    frame_count = int(config["frame_count"])
    total_events = int(round(float(level["event_rate_hz"]) * float(config["duration_s"])))
    margin = int(model.get("center_margin_px", 4))
    dry_centres = _eligible_centres(mask, False, margin)
    water_centres = _eligible_centres(mask, True, margin)
    if dry_centres.size == 0 and water_centres.size == 0:
        raise ValueError("No eligible event centre exists after applying image margin")
    if water_centres.size == 0:
        water_count = 0
    elif dry_centres.size == 0:
        water_count = total_events
    else:
        water_count = int(round(total_events * float(level["water_event_fraction"])))
    dry_count = total_events - water_count
    event_types = ["water_ripple"] * water_count + ["dry_splash"] * dry_count
    rng = np.random.default_rng(int(random_seed))
    rng.shuffle(event_types)
    events: list[dict[str, Any]] = []
    for index, event_type in enumerate(event_types):
        event_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        if event_type == "dry_splash":
            lifetime = _integer_range(rng, level["dry_lifetime_frames"])
            centres = dry_centres
        else:
            lifetime = _integer_range(rng, level["water_lifetime_frames"])
            centres = water_centres
        start = int(rng.integers(0, max(1, frame_count - lifetime + 1)))
        row, col = centres[int(rng.integers(0, centres.shape[0]))]
        common = {
            "event_id": f"event_{index:05d}",
            "event_type": event_type,
            "center_u": int(col),
            "center_v": int(row),
            "start_frame": start,
            "end_frame": start + lifetime - 1,
            "peak_frame": start + min(1, lifetime - 1),
            "random_seed": event_seed,
        }
        if event_type == "dry_splash":
            common.update({
                "radius_px": _integer_range(rng, level["dry_radius_px"]),
                "intensity": _float_range(rng, model["dry_intensity"]),
                "spoke_count": _integer_range(rng, model["dry_spoke_count"]),
                "satellite_count": _integer_range(rng, model["dry_satellite_count"]),
            })
        else:
            common.update({
                "initial_radius_px": _integer_range(rng, level["water_initial_radius_px"]),
                "expansion_rate_px_per_frame": _float_range(rng, level["water_expansion_rate_px_per_frame"]),
                "damping_factor": _float_range(rng, model["water_damping_factor"]),
                "ring_count": _integer_range(rng, model["water_ring_count"]),
                "ellipse_ratio": _float_range(rng, model["water_ellipse_ratio"]),
                "orientation_deg": float(rng.uniform(0.0, 180.0)),
                "intensity": _float_range(rng, model["water_intensity"]),
            })
        events.append(common)
    return sorted(events, key=lambda item: (item["start_frame"], item["event_id"]))


def _dry_event_patch(event: dict[str, Any], frame_index: int) -> tuple[np.ndarray, float]:
    age = frame_index - int(event["start_frame"])
    lifetime = int(event["end_frame"]) - int(event["start_frame"]) + 1
    phase = age / max(1, lifetime - 1)
    radius = int(event["radius_px"])
    patch_radius = radius + 3
    size = patch_radius * 2 + 1
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    centre = patch_radius
    rng = np.random.default_rng(int(event["random_seed"]))
    growth = 0.55 + 0.45 * min(1.0, age)
    current_radius = max(1.0, radius * growth)
    for _ in range(int(event["spoke_count"])):
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        length = current_radius * float(rng.uniform(0.55, 1.0))
        end = (centre + int(round(math.cos(angle) * length)), centre + int(round(math.sin(angle) * length)))
        draw.line((centre, centre, end[0], end[1]), fill=255, width=1)
    for _ in range(int(event["satellite_count"])):
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        distance = current_radius * float(rng.uniform(0.6, 1.15))
        x = centre + int(round(math.cos(angle) * distance))
        y = centre + int(round(math.sin(angle) * distance))
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=200)
    peak = 0.65 if age == 0 and lifetime > 1 else 1.0
    decay = math.exp(-2.6 * phase)
    amplitude = float(event["intensity"]) * peak * decay
    return np.asarray(canvas, dtype=np.float32) / 255.0, amplitude


def _ripple_patch(event: dict[str, Any], frame_index: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    age = frame_index - int(event["start_frame"])
    radius = float(event["initial_radius_px"]) + float(event["expansion_rate_px_per_frame"]) * age
    amplitude = float(event["intensity"]) * float(event["damping_factor"]) ** age
    patch_radius = int(math.ceil(radius + 5.0))
    coords = np.arange(-patch_radius, patch_radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    angle = math.radians(float(event["orientation_deg"]))
    rotated_x = math.cos(angle) * xx + math.sin(angle) * yy
    rotated_y = -math.sin(angle) * xx + math.cos(angle) * yy
    radial = np.sqrt(rotated_x * rotated_x + (rotated_y / float(event["ellipse_ratio"])) ** 2)
    field = np.zeros(radial.shape, dtype=np.float32)
    activity = np.zeros(radial.shape, dtype=bool)
    for ring_index in range(int(event["ring_count"])):
        ring_radius = radius - ring_index * 3.0
        if ring_radius <= 0.5:
            continue
        ring = np.exp(-0.5 * ((radial - ring_radius) / 0.85) ** 2)
        field += ring * (1.0 if ring_index % 2 == 0 else -0.65)
        activity |= np.abs(radial - ring_radius) <= 1.5
    if age <= 2:
        impact = np.exp(-0.5 * (radial / max(1.0, 2.2 - 0.4 * age)) ** 2)
        field += impact * (0.8 if age % 2 == 0 else -0.5)
        activity |= radial <= 2.5
    return field, activity, radius, amplitude


def _paste_delta(
    frame: np.ndarray,
    activity_map: np.ndarray,
    patch: np.ndarray,
    patch_activity: np.ndarray,
    center_u: int,
    center_v: int,
    amplitude: float,
    activity_bit: int,
    energy_weight: np.ndarray | None = None,
) -> None:
    patch_height, patch_width = patch.shape
    half_height, half_width = patch_height // 2, patch_width // 2
    y0, x0 = center_v - half_height, center_u - half_width
    y1, x1 = y0 + patch_height, x0 + patch_width
    image_y0, image_x0 = max(0, y0), max(0, x0)
    image_y1, image_x1 = min(frame.shape[0], y1), min(frame.shape[1], x1)
    if image_y0 >= image_y1 or image_x0 >= image_x1:
        return
    patch_y0, patch_x0 = image_y0 - y0, image_x0 - x0
    patch_y1, patch_x1 = patch_y0 + image_y1 - image_y0, patch_x0 + image_x1 - image_x0
    local = patch[patch_y0:patch_y1, patch_x0:patch_x1]
    if energy_weight is not None:
        local = local * energy_weight[image_y0:image_y1, image_x0:image_x1]
    frame[image_y0:image_y1, image_x0:image_x1, :] += local[..., None] * float(amplitude)
    active = patch_activity[patch_y0:patch_y1, patch_x0:patch_x1]
    target = activity_map[image_y0:image_y1, image_x0:image_x1]
    target[active] |= np.uint8(activity_bit)


def _shift_without_wrap(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return frame
    result = np.empty_like(frame)
    source_y0, source_y1 = max(0, -dy), min(frame.shape[0], frame.shape[0] - dy)
    source_x0, source_x1 = max(0, -dx), min(frame.shape[1], frame.shape[1] - dx)
    target_y0, target_y1 = source_y0 + dy, source_y1 + dy
    target_x0, target_x1 = source_x0 + dx, source_x1 + dx
    result[:] = frame[np.clip(np.arange(frame.shape[0]) - dy, 0, frame.shape[0] - 1)[:, None], np.clip(np.arange(frame.shape[1]) - dx, 0, frame.shape[1] - 1)[None, :]]
    result[target_y0:target_y1, target_x0:target_x1] = frame[source_y0:source_y1, source_x0:source_x1]
    return result


def _apply_environment(
    frame: np.ndarray,
    water_mask: np.ndarray,
    frame_index: int,
    config: dict[str, Any],
    level_config: dict[str, Any],
    random_seed: int,
) -> np.ndarray:
    environment = config["environment_perturbations"]
    fps = float(config["fps"])
    result = frame.astype(np.float32, copy=True)
    brightness = environment["global_brightness"]
    if brightness["enabled"]:
        phase = 2.0 * math.pi * frame_index / max(1.0, float(brightness["period_s"]) * fps)
        result *= 1.0 + float(brightness["amplitude"]) * math.sin(phase)
    exposure = environment["exposure_flicker"]
    if exposure["enabled"]:
        phase = 2.0 * math.pi * frame_index / max(1, int(exposure["period_frames"]))
        result += 255.0 * float(exposure["amplitude"]) * math.sin(phase)
    reflection = environment["water_reflection_flicker"]
    if reflection["enabled"] and np.any(water_mask):
        yy, xx = np.indices(water_mask.shape)
        texture = np.sin((xx + frame_index * 1.7) / float(reflection["spatial_period_px"]) * 2.0 * math.pi)
        result += texture[..., None] * water_mask[..., None] * float(reflection["intensity"])
    rng = np.random.default_rng(int(random_seed) + frame_index * 104729)
    streaks = environment["rain_streaks"]
    if streaks["enabled"] and int(level_config["rain_streaks_per_frame"]) > 0:
        overlay = Image.new("L", (result.shape[1], result.shape[0]), 0)
        draw = ImageDraw.Draw(overlay)
        for _ in range(int(level_config["rain_streaks_per_frame"])):
            x = int(rng.integers(0, result.shape[1]))
            y = int(rng.integers(0, result.shape[0]))
            length = _integer_range(rng, streaks["length_px"])
            draw.line((x, y, x + 1, min(result.shape[0] - 1, y + length)), fill=255, width=1)
        result += np.asarray(overlay, dtype=np.float32)[..., None] / 255.0 * float(streaks["intensity"])
    noise = environment["gaussian_noise"]
    if noise["enabled"] and float(level_config["gaussian_noise_sigma"]) > 0:
        result += rng.normal(0.0, float(level_config["gaussian_noise_sigma"]), result.shape).astype(np.float32)
    compression = environment["compression_artifact"]
    if compression["enabled"]:
        clipped = np.clip(result, 0, 255).astype(np.uint8)
        image = Image.fromarray(clipped, mode="RGB")
        factor = int(compression["downscale_factor"])
        small = image.resize((max(1, image.width // factor), max(1, image.height // factor)), _RESAMPLING.BILINEAR)
        blocky = np.asarray(small.resize(image.size, _RESAMPLING.NEAREST), dtype=np.float32)
        blend = float(compression["blend"])
        result = result * (1.0 - blend) + blocky * blend
    jitter = environment["camera_jitter"]
    if jitter["enabled"] and int(jitter["max_offset_px"]) > 0:
        maximum = int(jitter["max_offset_px"])
        result = _shift_without_wrap(result, int(rng.integers(-maximum, maximum + 1)), int(rng.integers(-maximum, maximum + 1)))
    return np.clip(result, 0, 255).astype(np.uint8)


def _static_water_appearance(base_rgb: np.ndarray, water_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask_image = Image.fromarray(np.where(water_mask, 255, 0).astype(np.uint8), mode="L")
    soft = np.asarray(mask_image.filter(ImageFilter.GaussianBlur(radius=2.5)), dtype=np.float32) / 255.0
    frame = base_rgb.astype(np.float32).copy()
    frame *= 1.0 - soft[..., None] * 0.10
    frame[..., 2] += soft * 7.0
    return np.clip(frame, 0, 255).astype(np.uint8), soft


def _git_info(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(["git", "status", "--porcelain"], cwd=project_root, text=True, capture_output=True, check=True).stdout.strip()
    )
    return commit, dirty


def _camera_boundary_touch(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    border = np.zeros(binary.shape, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    return {
        "touches_image_boundary": bool(np.any(binary & border)),
        "boundary_water_pixel_count": int(np.count_nonzero(binary & border)),
    }


def generate_sequence(
    base_rgb: np.ndarray,
    water_mask: np.ndarray,
    config: dict[str, Any],
    case_id: str,
    rain_level: str,
    random_seed: int,
    output_dir: str | Path,
    base_image_source: dict[str, Any],
    project_root: str | Path,
    create_preview: bool = True,
) -> dict[str, Any]:
    base = np.asarray(base_rgb, dtype=np.uint8)
    mask = np.asarray(water_mask, dtype=bool)
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("base_rgb must have shape HxWx3")
    if mask.shape != base.shape[:2]:
        raise ValueError(f"water_mask shape {mask.shape} does not match base image {base.shape[:2]}")
    if rain_level not in config["rain_levels"]:
        raise KeyError(f"Unknown rain_level: {rain_level}")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    frames_dir = output / "frames"
    gt_dir = output / "ground_truth"
    metadata_dir = output / "metadata"
    frames_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    events = generate_event_schedule(mask, config, rain_level, random_seed)
    repeated_events = generate_event_schedule(mask, config, rain_level, random_seed)
    alternate_events = generate_event_schedule(mask, config, rain_level, random_seed + 1)
    frame_count = int(config["frame_count"])
    level_config = config["rain_levels"][rain_level]
    static_base, soft_water = _static_water_appearance(base, mask)
    outside_fraction = float(config["event_model"]["water_energy_outside_mask_fraction"])
    water_energy = outside_fraction + (1.0 - outside_fraction) * soft_water
    event_map = np.zeros((frame_count, mask.shape[0], mask.shape[1]), dtype=np.uint8)
    temporal_activity = np.zeros(mask.shape, dtype=np.uint16)
    per_frame_counts: list[dict[str, int]] = []
    event_states: list[dict[str, Any]] = []
    sequence_hash = hashlib.sha256()
    frame_paths: list[Path] = []
    all_frames_finite = True
    dataset_pixel_min = 255
    dataset_pixel_max = 0
    for frame_index in range(frame_count):
        frame = static_base.astype(np.float32)
        active_dry = 0
        active_water = 0
        states: list[dict[str, Any]] = []
        for event in events:
            if not (int(event["start_frame"]) <= frame_index <= int(event["end_frame"])):
                continue
            if event["event_type"] == "dry_splash":
                patch, amplitude = _dry_event_patch(event, frame_index)
                _paste_delta(
                    frame,
                    event_map[frame_index],
                    patch,
                    patch > 0.08,
                    int(event["center_u"]),
                    int(event["center_v"]),
                    amplitude,
                    1,
                )
                active_dry += 1
                states.append({
                    "event_id": event["event_id"],
                    "event_type": "dry_splash",
                    "center_u": event["center_u"],
                    "center_v": event["center_v"],
                    "radius_px": event["radius_px"],
                    "amplitude": amplitude,
                })
            else:
                patch, activity, radius, amplitude = _ripple_patch(event, frame_index)
                _paste_delta(
                    frame,
                    event_map[frame_index],
                    patch,
                    activity,
                    int(event["center_u"]),
                    int(event["center_v"]),
                    amplitude,
                    2,
                    water_energy,
                )
                active_water += 1
                states.append({
                    "event_id": event["event_id"],
                    "event_type": "water_ripple",
                    "center_u": event["center_u"],
                    "center_v": event["center_v"],
                    "radius_px": radius,
                    "amplitude": amplitude,
                })
        frame = _apply_environment(frame, mask, frame_index, config, level_config, random_seed)
        all_frames_finite = bool(all_frames_finite and np.isfinite(frame).all())
        dataset_pixel_min = min(dataset_pixel_min, int(np.min(frame)))
        dataset_pixel_max = max(dataset_pixel_max, int(np.max(frame)))
        path = frames_dir / f"frame_{frame_index:06d}.png"
        Image.fromarray(frame, mode="RGB").save(path, compress_level=3)
        frame_paths.append(path)
        sequence_hash.update(path.read_bytes())
        temporal_activity += (event_map[frame_index] > 0).astype(np.uint16)
        per_frame_counts.append({"frame_index": frame_index, "dry_splash": active_dry, "water_ripple": active_water})
        event_states.append({"frame_index": frame_index, "active_events": states})

    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(gt_dir / "water_mask.png")
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(gt_dir / "observable_water_mask.png")
    np.save(gt_dir / "event_map_sequence.npy", event_map)
    np.save(gt_dir / "temporal_activity_map.npy", temporal_activity)
    with (gt_dir / "event_annotations.json").open("w", encoding="utf-8") as stream:
        json.dump({"data_role": "synthetic_ground_truth", "event_map_bits": {"dry_splash": 1, "water_ripple": 2}, "events": events}, stream, indent=2)
        stream.write("\n")
    with (gt_dir / "frame_event_counts.json").open("w", encoding="utf-8") as stream:
        json.dump(per_frame_counts, stream, indent=2)
        stream.write("\n")
    with (gt_dir / "event_states.json").open("w", encoding="utf-8") as stream:
        json.dump(event_states, stream, indent=2)
        stream.write("\n")

    import yaml

    with (metadata_dir / "generator_config_snapshot.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    dry_events = [event for event in events if event["event_type"] == "dry_splash"]
    water_events = [event for event in events if event["event_type"] == "water_ripple"]
    dry_lifetimes = [event["end_frame"] - event["start_frame"] + 1 for event in dry_events]
    water_lifetimes = [event["end_frame"] - event["start_frame"] + 1 for event in water_events]
    required_gt = [
        gt_dir / "water_mask.png",
        gt_dir / "observable_water_mask.png",
        gt_dir / "event_map_sequence.npy",
        gt_dir / "temporal_activity_map.npy",
        gt_dir / "event_annotations.json",
        gt_dir / "frame_event_counts.json",
        gt_dir / "event_states.json",
        metadata_dir / "generator_config_snapshot.yaml",
    ]
    frame_names = [path.name for path in frame_paths]
    expected_names = [f"frame_{index:06d}.png" for index in range(frame_count)]
    checks = {
        "frame_count_correct": len(frame_paths) == frame_count,
        "frame_numbering_contiguous": frame_names == expected_names,
        "frame_dimensions_consistent": all(Image.open(path).size == (base.shape[1], base.shape[0]) for path in frame_paths),
        "frames_no_nan_inf": all_frames_finite,
        "frame_values_uint8_range": 0 <= dataset_pixel_min <= dataset_pixel_max <= 255,
        "dry_centres_outside_water": all(not mask[event["center_v"], event["center_u"]] for event in dry_events),
        "water_centres_inside_water": all(mask[event["center_v"], event["center_u"]] for event in water_events),
        "dry_scene_has_no_water_events": bool(np.any(mask)) or len(water_events) == 0,
        "dry_lifetime_shorter_than_water": not dry_lifetimes or not water_lifetimes or float(np.mean(dry_lifetimes)) < float(np.mean(water_lifetimes)),
        "water_radius_expands_and_amplitude_decays": all(event["expansion_rate_px_per_frame"] > 0 and 0 < event["damping_factor"] < 1 for event in water_events),
        "same_seed_schedule_identical": events == repeated_events,
        "different_seed_schedule_differs": events != alternate_events,
        "rain_event_rate_order_valid": float(config["rain_levels"]["light"]["event_rate_hz"]) < float(config["rain_levels"]["moderate"]["event_rate_hz"]) < float(config["rain_levels"]["heavy"]["event_rate_hz"]),
        "ground_truth_files_complete": all(path.is_file() for path in required_gt),
        "base_image_resolution_matches": base.shape[:2] == mask.shape,
        "detector_frames_directory_has_no_ground_truth_links": not any(path.is_symlink() for path in frames_dir.iterdir()),
    }
    quality = {
        "status": "pass" if all(checks.values()) else "reject",
        "checks": checks,
        "metrics": {
            "dry_event_count": len(dry_events),
            "water_event_count": len(water_events),
            "mean_dry_lifetime_frames": float(np.mean(dry_lifetimes)) if dry_lifetimes else None,
            "mean_water_lifetime_frames": float(np.mean(water_lifetimes)) if water_lifetimes else None,
            "sequence_sha256": sequence_hash.hexdigest(),
            "dataset_pixel_min": dataset_pixel_min,
            "dataset_pixel_max": dataset_pixel_max,
            "camera_boundary": _camera_boundary_touch(mask),
        },
        "ground_truth_used_for_quality_only": True,
    }
    write_json_path = metadata_dir / "generation_quality_report.json"
    with write_json_path.open("w", encoding="utf-8") as stream:
        json.dump(quality, stream, indent=2)
        stream.write("\n")

    commit, dirty = _git_info(Path(project_root).resolve())
    manifest = {
        "schema_version": str(config["schema_version"]),
        "data_role": "synthetic_training_data",
        "case_id": case_id,
        "rain_level": rain_level,
        "fps": int(config["fps"]),
        "duration_s": float(config["duration_s"]),
        "frame_count": frame_count,
        "width": int(base.shape[1]),
        "height": int(base.shape[0]),
        "random_seed": int(random_seed),
        "base_image_source": base_image_source,
        "generator_version": str(config["generator_version"]),
        "generator_source": GENERATOR_SOURCE,
        "ground_truth_water_mask_source": f"data/simulation/{case_id}/ground_truth/camera_water_mask_gt.png",
        "dry_event_count": len(dry_events),
        "water_event_count": len(water_events),
        "environment_perturbations": config["environment_perturbations"],
        "created_at": str(config["deterministic_created_at"]),
        "git_commit": commit,
        "git_dirty": dirty,
        "ground_truth_used_for_generation_only": True,
        "detector_input_should_only_use_frames": True,
        "synthetic_physics_validity": "visual_abstraction_not_fluid_simulation",
        "output_layout": {"detector_input": "frames/", "labels": "ground_truth/", "metadata": "metadata/"},
        "camera_boundary": _camera_boundary_touch(mask),
        "sequence_sha256": sequence_hash.hexdigest(),
        "generation_quality_status": quality["status"],
    }
    with (metadata_dir / "sequence_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    if create_preview:
        stride = max(1, int(config.get("preview_stride_frames", 10)))
        preview_frames = [Image.open(path).convert("RGB") for path in frame_paths[::stride]]
        preview_frames[0].save(
            output / "preview.gif",
            save_all=True,
            append_images=preview_frames[1:],
            duration=int(round(1000.0 * stride / float(config["fps"]))),
            loop=0,
        )
    return {"manifest": manifest, "quality": quality, "events": events, "output_dir": str(output)}
