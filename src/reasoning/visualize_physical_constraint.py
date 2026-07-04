#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S7-C: Visualize simplified physical constraint check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_physical_constraint_figure(
    physical_result: dict[str, Any],
    final_result: dict[str, Any],
    output_path: Path,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate S7-C figure: {exc}") from exc

    physical_items = sorted(physical_result["physical_check_results"], key=lambda item: int(item["horizon_min"]))
    horizons = [int(item["horizon_min"]) for item in physical_items]
    corrected_depths = [float(item["corrected_depth_cm"]) for item in physical_items]
    adjusted_depths = [float(item["adjusted_depth_cm"]) for item in physical_items]
    error_ratios = [float(item["error_ratio"]) for item in physical_items]
    confidences = [str(item["physical_confidence"]) for item in physical_items]
    confidence_colors = {"high": "#22c55e", "medium": "#ca8a04", "low": "#dc2626"}
    bar_colors = [confidence_colors.get(value, "#64748b") for value in confidences]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 8), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.45, 1.35, 1.0])

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.set_axis_off()
    title_ax.text(
        0.5,
        0.72,
        "S7-C Physical Constraint Check - MVP",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    title_ax.text(
        0.5,
        0.25,
        "simplified_water_balance_mvp, not full hydrodynamic model",
        ha="center",
        va="center",
        fontsize=10,
        color="#b45309",
    )

    forecast_ax = fig.add_subplot(grid[1, :])
    forecast_ax.plot(horizons, corrected_depths, marker="o", linewidth=2.2, color="#64748b", label="S7-B corrected")
    forecast_ax.plot(horizons, adjusted_depths, marker="o", linewidth=2.5, color="#0284c7", label="S7-C adjusted/final")
    forecast_ax.axhline(15, color="#2563eb", linestyle="--", linewidth=1.0, label="blue 15 cm")
    forecast_ax.axhline(30, color="#ca8a04", linestyle="--", linewidth=1.0, label="yellow 30 cm")
    forecast_ax.axhline(50, color="#ea580c", linestyle="--", linewidth=1.0, label="orange 50 cm")
    forecast_ax.set_title("Forecast depth after physical constraint")
    forecast_ax.set_xlabel("horizon (min)")
    forecast_ax.set_ylabel("forecast depth (cm)")
    forecast_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    forecast_ax.legend(fontsize=8)
    for x_value, y_value, confidence in zip(horizons, adjusted_depths, confidences):
        forecast_ax.text(
            x_value,
            y_value + 2.0,
            f"{y_value:.1f} cm\n{confidence}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#0f172a",
        )

    ratio_ax = fig.add_subplot(grid[2, 0])
    ratio_ax.bar([str(h) for h in horizons], error_ratios, color=bar_colors)
    ratio_ax.axhline(float(physical_result["tolerance_ratio"]), color="#dc2626", linestyle="--", linewidth=1.0)
    ratio_ax.set_title("Water-balance error ratio")
    ratio_ax.set_xlabel("horizon (min)")
    ratio_ax.set_ylabel("error ratio")
    ratio_ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    summary_ax = fig.add_subplot(grid[2, 1])
    summary_ax.set_axis_off()
    summary_ax.text(0.02, 0.82, "Final forecast summary", fontsize=13, fontweight="bold", color="#0f172a")
    summary_ax.text(0.02, 0.62, f"Overall warning: {final_result['overall_warning_level']}", fontsize=11, color="#0f172a")
    summary_ax.text(
        0.02,
        0.44,
        "Physical confidence: "
        f"{final_result['physical_confidence_summary']['overall_physical_confidence']}",
        fontsize=11,
        color="#0f172a",
    )
    summary_ax.text(
        0.02,
        0.24,
        "Model: linear volume-depth proxy\n"
        "Inputs: current volume, rainfall, drainage, infiltration",
        fontsize=9,
        color="#475569",
    )

    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def visualize_physical_constraint(config_path: str | Path, project_root: str | Path) -> dict[str, str]:
    del config_path
    root = Path(project_root).expanduser().resolve()
    physical_path = root / "outputs" / "json" / "physical_constraint_result.json"
    final_path = root / "outputs" / "json" / "final_forecast_result.json"
    output_path = root / "outputs" / "figures" / "physical_constraint_summary.png"
    physical_result = load_json(physical_path)
    final_result = load_json(final_path)
    save_physical_constraint_figure(physical_result, final_result, output_path)
    print(f"[S7-C][physical_figure] output file: {output_path}")
    return {"physical_constraint_summary": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate S7-C physical constraint summary figure.")
    parser.add_argument("--config", required=True, help="Path to configs/physical_constraint_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    visualize_physical_constraint(args.config, args.project_root)


if __name__ == "__main__":
    main()
