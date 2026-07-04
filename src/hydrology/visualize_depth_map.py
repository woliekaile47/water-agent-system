#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualization helper for S4 water depth maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_depth_heatmap(
    depth_map_m: np.ndarray,
    valid_mask: np.ndarray,
    output_path: str | Path,
    dem_metadata: dict[str, Any],
    title: str = "S4 Water Depth Map - Region-level MVP",
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"缺少 matplotlib，无法生成水深热力图: {exc}") from exc

    roi = dem_metadata["dem_roi"]
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    depth_cm = depth_map_m * 100.0
    masked_depth = np.ma.masked_where(~valid_mask, depth_cm)
    vmax = float(np.max(depth_cm[valid_mask])) if np.any(valid_mask) else 1.0
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    image = ax.imshow(
        masked_depth,
        origin="lower",
        extent=[roi["x_min"], roi["x_max"], roi["y_min"], roi["y_max"]],
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("water depth (cm)")
    ax.set_title(title + " (configured_mvp_simulation)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Visualize S4 water depth map.")
    parser.add_argument("--depth_map", required=True)
    parser.add_argument("--valid_mask", required=True)
    parser.add_argument("--dem_metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    depth_map = np.load(Path(args.depth_map).expanduser())
    valid_mask = np.load(Path(args.valid_mask).expanduser()).astype(bool)
    with Path(args.dem_metadata).expanduser().open("r", encoding="utf-8") as f:
        dem_metadata = json.load(f)
    output = save_depth_heatmap(depth_map, valid_mask, args.output, dem_metadata)
    print(f"[S4][visualize_depth] saved: {output}")


if __name__ == "__main__":
    main()
