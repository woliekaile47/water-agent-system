#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a DEM-space water-region mask from row/column polygon points."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(f"PyYAML is required to read config: {exc}") from exc

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "dem_space_water_mask" not in data:
        raise ValueError("Config must contain top-level 'dem_space_water_mask'.")
    return data["dem_space_water_mask"]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def get_grid_resolution(metadata: dict[str, Any]) -> float:
    value = (
        metadata.get("grid_size")
        or metadata.get("grid_resolution")
        or metadata.get("resolution")
        or metadata.get("cell_size")
        or metadata.get("cell_size_m")
    )
    if value is None:
        raise ValueError("ground DEM metadata is missing grid_size/grid_resolution/resolution.")
    return float(value)


def validate_polygon(points_rc: list[Any], dem_shape: tuple[int, int]) -> np.ndarray:
    if len(points_rc) < 3:
        raise ValueError("polygon_points_rc must contain at least 3 points.")
    polygon = np.asarray(points_rc, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon_points_rc must be a list of [row, col] pairs.")
    rows = polygon[:, 0]
    cols = polygon[:, 1]
    if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= dem_shape[0]) or np.any(cols >= dem_shape[1]):
        raise ValueError(
            "polygon_points_rc contains points outside the DEM shape "
            f"{dem_shape}: {points_rc}"
        )
    return polygon


def rasterize_polygon_rc(dem_shape: tuple[int, int], polygon_rc: np.ndarray) -> np.ndarray:
    try:
        from matplotlib.path import Path as MplPath
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for polygon rasterization: {exc}") from exc

    rows, cols = np.indices(dem_shape, dtype=np.float64)
    # matplotlib Path uses x/y ordering. Here x=col center, y=row center.
    points_xy = np.column_stack(((cols.ravel() + 0.5), (rows.ravel() + 0.5)))
    polygon_xy = np.column_stack((polygon_rc[:, 1], polygon_rc[:, 0]))
    path = MplPath(polygon_xy)
    return path.contains_points(points_xy).reshape(dem_shape)


def extract_boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    all_neighbors_in_region = np.ones(mask.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
            all_neighbors_in_region &= neighbor
    return mask & ~all_neighbors_in_region


def save_mask_png(mask: np.ndarray, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to save mask png: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, mask.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)


def save_overlay(
    base_array: np.ndarray,
    base_valid_mask: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate overlay figure: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    boundary = extract_boundary_mask(mask)
    masked_base = np.ma.masked_where(~base_valid_mask, base_array)
    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    image = ax.imshow(masked_base, origin="lower", cmap=cmap)
    rows, cols = np.where(boundary)
    ax.scatter(cols, rows, s=2, c="#ef4444", marker="s", linewidths=0, label="mask boundary")
    ax.set_title(title)
    ax.set_xlabel("grid col")
    ax.set_ylabel("grid row")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def create_dem_space_water_mask(
    config_path: str | Path,
    project_root: str | Path,
    use_refined: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)

    ground_dem_path = resolve_project_path(root, config["ground_dem_interpolated_path"])
    ground_valid_path = resolve_project_path(root, config["ground_dem_valid_mask_path"])
    point_count_path = resolve_project_path(root, config["ground_dem_point_count_path"])
    metadata_path = resolve_project_path(root, config["ground_dem_metadata_path"])

    ground_dem = np.load(ground_dem_path)
    ground_valid_mask = np.load(ground_valid_path).astype(bool)
    point_count = np.load(point_count_path)
    metadata = load_json(metadata_path)
    dem_shape = tuple(int(v) for v in ground_dem.shape)
    if ground_valid_mask.shape != dem_shape or point_count.shape != dem_shape:
        raise ValueError(
            "ground_dem, ground_valid_mask, and ground_dem_point_count shapes must match: "
            f"{dem_shape}, {ground_valid_mask.shape}, {point_count.shape}"
        )

    polygon_field = "refined_polygon_points_rc" if use_refined else "polygon_points_rc"
    if polygon_field not in config:
        raise ValueError(f"Config is missing {polygon_field}.")
    polygon_rc = validate_polygon(config[polygon_field], dem_shape)
    mask = rasterize_polygon_rc(dem_shape, polygon_rc)
    mask_cell_count = int(np.count_nonzero(mask))
    if mask_cell_count == 0:
        raise RuntimeError("Polygon produced zero mask cells. Refuse to write an empty water-region mask.")

    grid_resolution = get_grid_resolution(metadata)
    mask_area_m2 = float(mask_cell_count * grid_resolution * grid_resolution)

    output_mask_npy = resolve_project_path(root, config["output_mask_npy"])
    output_mask_png = resolve_project_path(root, config["output_mask_png"])
    output_metadata = resolve_project_path(root, config["output_metadata"])
    output_dem_overlay = resolve_project_path(root, config["output_dem_overlay"])
    output_point_count_overlay = resolve_project_path(root, config["output_point_count_overlay"])

    output_mask_npy.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_mask_npy, mask.astype(bool))
    save_mask_png(mask, output_mask_png)
    save_overlay(
        base_array=ground_dem.astype(np.float32),
        base_valid_mask=np.isfinite(ground_dem) & ground_valid_mask,
        mask=mask,
        output_path=output_dem_overlay,
        title=f"DEM-space Water Region Mask on Ground DEM - {config['scene_type']}",
        colorbar_label="ground z (m)",
        cmap="viridis",
    )
    save_overlay(
        base_array=point_count.astype(np.float32),
        base_valid_mask=point_count > 0,
        mask=mask,
        output_path=output_point_count_overlay,
        title=f"DEM-space Water Region Mask on Point Count - {config['scene_type']}",
        colorbar_label="points / cell",
        cmap="magma",
    )

    result = {
        "stage": "S3_playground_pit_dem_space_water_mask",
        "scene_type": str(config["scene_type"]),
        "case_name": str(config["case_name"]),
        "dem_shape": list(dem_shape),
        "mask_shape": list(mask.shape),
        "polygon_points_rc": [[int(r), int(c)] for r, c in polygon_rc],
        "polygon_source": polygon_field,
        "mask_cell_count": mask_cell_count,
        "mask_area_m2": mask_area_m2,
        "grid_resolution": grid_resolution,
        "source_ground_dem": str(ground_dem_path),
        "source_ground_valid_mask": str(ground_valid_path),
        "source_ground_point_count": str(point_count_path),
        "source_ground_metadata": str(metadata_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": config.get("note", ""),
        "output_files": {
            "water_region_mask_npy": str(output_mask_npy),
            "water_region_mask_png": str(output_mask_png),
            "water_region_mask_metadata": str(output_metadata),
            "water_region_mask_on_dem": str(output_dem_overlay),
            "water_region_mask_on_point_count": str(output_point_count_overlay),
        },
    }
    with output_metadata.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[dem_mask] scene_type: {result['scene_type']}")
    print(f"[dem_mask] case_name: {result['case_name']}")
    print(f"[dem_mask] ground DEM shape: {result['dem_shape']}")
    print(f"[dem_mask] mask shape: {result['mask_shape']}")
    print(f"[dem_mask] mask_cell_count: {mask_cell_count}")
    print(f"[dem_mask] mask_area_m2: {mask_area_m2:.4f}")
    print("[dem_mask] output paths:")
    for path in result["output_files"].values():
        print(f"  - {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a DEM-space water-region mask from polygon row/col points.")
    parser.add_argument("--config", required=True, help="Path to DEM-space mask YAML config")
    parser.add_argument("--use-refined", action="store_true", help="Use refined_polygon_points_rc instead of polygon_points_rc")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    create_dem_space_water_mask(args.config, args.project_root, args.use_refined)


if __name__ == "__main__":
    main()
