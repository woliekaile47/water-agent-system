#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S5: Visualize area and volume summary metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Area/volume result JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_area_volume_summary_figure(result: dict[str, Any], output_path: Path) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate the S5 summary figure: {exc}") from exc

    metrics = [
        ("Water Area", f"{float(result.get('water_area_m2', 0.0)):.3f} m2"),
        ("Water Volume", f"{float(result.get('water_volume_m3', 0.0)):.3f} m3\n{float(result.get('water_volume_liter', 0.0)):.1f} L"),
        ("Max Depth", f"{float(result.get('max_depth_cm', 0.0)):.2f} cm"),
        ("Mean Depth", f"{float(result.get('mean_depth_cm', 0.0)):.2f} cm"),
        ("Valid Cells", f"{int(result.get('valid_depth_cell_count', 0))}"),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.set_axis_off()
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    ax.text(
        0.5,
        0.90,
        "S5 Area & Volume Summary - MVP",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.80,
        "configured_mvp_simulation, not final real water-depth measurement",
        ha="center",
        va="center",
        fontsize=10,
        color="#b45309",
        transform=ax.transAxes,
    )

    for index, (label, value) in enumerate(metrics):
        x = 0.11 + index * 0.195
        rect = plt.Rectangle(
            (x - 0.085, 0.30),
            0.17,
            0.34,
            transform=ax.transAxes,
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            0.56,
            label,
            ha="center",
            va="center",
            fontsize=11,
            color="#475569",
            transform=ax.transAxes,
        )
        ax.text(
            x,
            0.42,
            value,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="#0369a1",
            transform=ax.transAxes,
        )

    timestamp = result.get("timestamp", "")
    ax.text(
        0.5,
        0.12,
        f"Generated from S4 depth method: {result.get('depth_method_from_s4')} | {timestamp}",
        ha="center",
        va="center",
        fontsize=9,
        color="#64748b",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def visualize_area_volume_summary(config_path: str | Path, project_root: str | Path) -> dict[str, str]:
    del config_path
    root = Path(project_root).expanduser().resolve()
    result_json = root / "outputs" / "json" / "water_area_volume_result.json"
    output_figure = root / "outputs" / "figures" / "water_area_volume_summary.png"
    result = _load_json(result_json)
    save_area_volume_summary_figure(result, output_figure)
    print(f"[S5][summary] output file: {output_figure}")
    return {"water_area_volume_summary": str(output_figure)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate S5 area and volume summary figure.")
    parser.add_argument("--config", required=True, help="Path to configs/roi_mapping.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    visualize_area_volume_summary(args.config, args.project_root)


if __name__ == "__main__":
    main()
