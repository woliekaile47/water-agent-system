#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4: Region-level manual mapping from camera mask to DEM grid."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(f"缺少 PyYAML，无法读取配置: {exc}") from exc
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "roi_mapping" not in data:
        raise ValueError("配置文件缺少 roi_mapping 字段")
    return data["roi_mapping"]


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_grid_range(
    dem_water_roi: dict[str, float],
    dem_roi: dict[str, float],
    grid_size: float,
    dem_shape: list[int],
) -> dict[str, int]:
    x_min = float(dem_water_roi["x_min"])
    x_max = float(dem_water_roi["x_max"])
    y_min = float(dem_water_roi["y_min"])
    y_max = float(dem_water_roi["y_max"])
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(f"dem_water_roi 范围异常: {dem_water_roi}")
    if (
        x_min < float(dem_roi["x_min"])
        or x_max > float(dem_roi["x_max"])
        or y_min < float(dem_roi["y_min"])
        or y_max > float(dem_roi["y_max"])
    ):
        raise ValueError(
            "dem_water_roi 超出 DEM 范围，请人工调整 configs/roi_mapping.yaml。"
            f" dem_water_roi={dem_water_roi}, dem_roi={dem_roi}"
        )

    ny, nx = int(dem_shape[0]), int(dem_shape[1])
    ix_min = int(math.floor((x_min - float(dem_roi["x_min"])) / grid_size))
    ix_max = int(math.ceil((x_max - float(dem_roi["x_min"])) / grid_size)) - 1
    iy_min = int(math.floor((y_min - float(dem_roi["y_min"])) / grid_size))
    iy_max = int(math.ceil((y_max - float(dem_roi["y_min"])) / grid_size)) - 1
    ix_min = max(0, min(nx - 1, ix_min))
    ix_max = max(0, min(nx - 1, ix_max))
    iy_min = max(0, min(ny - 1, iy_min))
    iy_max = max(0, min(ny - 1, iy_max))
    if ix_min > ix_max or iy_min > iy_max:
        raise ValueError("dem_water_roi 映射到 DEM 后没有有效栅格，请调整 configs/roi_mapping.yaml")
    return {
        "ix_min": ix_min,
        "ix_max": ix_max,
        "iy_min": iy_min,
        "iy_max": iy_max,
    }


def save_water_region_figure(
    water_region_mask: np.ndarray,
    dem_metadata: dict[str, Any],
    output_path: Path,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"缺少 matplotlib，无法生成 water_region_on_dem 图: {exc}") from exc

    roi = dem_metadata["dem_roi"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    image = ax.imshow(
        water_region_mask.astype(np.uint8),
        origin="lower",
        extent=[roi["x_min"], roi["x_max"], roi["y_min"], roi["y_max"]],
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("water region mask")
    ax.set_title("S4 Region-level Water Mask on DEM Grid")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def map_mask_to_dem(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    config = load_yaml(config_path)
    mapping_method = str(config.get("mapping_method", "region_level_manual"))
    if mapping_method != "region_level_manual":
        raise ValueError(f"S4 MVP 当前只支持 region_level_manual，不支持: {mapping_method}")

    mask_path = resolve_project_path(root, config["camera_mask"]["mask_path"])
    mask_metadata_path = resolve_project_path(root, config["camera_mask"]["metadata_path"])
    dem_metadata_path = resolve_project_path(root, config["dem"]["metadata_path"])
    if not mask_path.exists():
        raise FileNotFoundError(f"manual mask 不存在: {mask_path}。请先运行 --stage manual_mask。")
    if not mask_metadata_path.exists():
        raise FileNotFoundError(f"manual mask metadata 不存在: {mask_metadata_path}")

    camera_mask = np.load(mask_path)
    mask_metadata = load_json(mask_metadata_path)
    dem_metadata = load_json(dem_metadata_path)
    grid_size = float(dem_metadata["grid_size"])
    dem_shape = dem_metadata["dem_shape"]
    dem_roi = dem_metadata["dem_roi"]
    grid_range = compute_grid_range(config["dem_water_roi"], dem_roi, grid_size, dem_shape)

    water_region_mask = np.zeros((int(dem_shape[0]), int(dem_shape[1])), dtype=bool)
    water_region_mask[
        grid_range["iy_min"] : grid_range["iy_max"] + 1,
        grid_range["ix_min"] : grid_range["ix_max"] + 1,
    ] = True
    water_region_cell_count = int(np.count_nonzero(water_region_mask))
    if water_region_cell_count == 0:
        raise RuntimeError("water_region_mask 为空，请人工调整 configs/roi_mapping.yaml")

    fusion_dir = root / "data" / "fusion"
    figure_dir = root / "outputs" / "figures"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_mask = fusion_dir / "water_region_mask.npy"
    output_json = fusion_dir / "mask_to_dem_mapping.json"
    output_figure = figure_dir / "water_region_on_dem.png"

    np.save(output_mask, water_region_mask)
    save_water_region_figure(water_region_mask, dem_metadata, output_figure)

    result = {
        "stage": "S4_mask_to_dem_mapping",
        "mapping_method": mapping_method,
        "camera_mask_path": str(mask_path),
        "camera_mask_shape": list(camera_mask.shape),
        "camera_mask_pixel_count": int(mask_metadata.get("mask_pixel_count", int(np.count_nonzero(camera_mask)))),
        "dem_water_roi": config["dem_water_roi"],
        "dem_grid_index_range": grid_range,
        "grid_size": grid_size,
        "water_region_cell_count": water_region_cell_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Region-level manual mapping, not pixel-level calibration.",
        "output_files": {
            "water_region_mask": str(output_mask),
            "mask_to_dem_mapping": str(output_json),
            "water_region_on_dem": str(output_figure),
        },
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[S4][mask_to_dem] water region cell count: {water_region_cell_count}")
    print("[S4][mask_to_dem] output files:")
    for path in result["output_files"].values():
        print(f"  - {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Map S3 camera mask to DEM grid using manual region-level ROI.")
    parser.add_argument("--config", required=True, help="Path to configs/roi_mapping.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    map_mask_to_dem(args.config, args.project_root)


if __name__ == "__main__":
    main()
