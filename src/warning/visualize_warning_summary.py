#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S8: Visualize warning summary."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.warning.generate_warning_decision import load_warning_config, resolve_project_path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_warning_summary(decision: dict[str, Any], output_path: Path) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate S8 warning summary: {exc}") from exc

    forecast_results = sorted(decision["forecast_warning_results"], key=lambda item: int(item["horizon_min"]))
    horizons = [f"{int(item['horizon_min'])} min" for item in forecast_results]
    depths = [float(item["forecast_depth_cm"]) for item in forecast_results]
    levels = [str(item["warning_level"]) for item in forecast_results]
    level_colors = {
        "none": "#94a3b8",
        "blue": "#2563eb",
        "yellow": "#ca8a04",
        "orange": "#ea580c",
    }
    colors = [level_colors.get(level, "#64748b") for level in levels]
    orange_time = decision.get("time_to_thresholds_min", {}).get("orange")
    orange_time_text = "N/A" if orange_time is None else f"{float(orange_time):.2f} min"
    action_short = textwrap.shorten(decision["action_suggestion"], width=155, placeholder="...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 7), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    grid = fig.add_gridspec(3, 4, height_ratios=[0.75, 1.4, 0.9], hspace=0.25)

    header = fig.add_subplot(grid[0, :])
    header.set_axis_off()
    header.text(
        0.5,
        0.78,
        "S8 Warning Summary - MVP",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    header.text(
        0.5,
        0.42,
        "MVP simulation, not final emergency dispatch advice",
        ha="center",
        va="center",
        fontsize=10,
        color="#b45309",
    )
    header.text(
        0.5,
        0.10,
        f"Overall warning level: {decision['overall_warning_level'].upper()} | "
        f"Current mean depth: {decision['current_mean_depth_cm']:.2f} cm | "
        f"Time to orange: {orange_time_text}",
        ha="center",
        va="center",
        fontsize=12,
        color="#0f172a",
    )

    chart_ax = fig.add_subplot(grid[1, :])
    bars = chart_ax.bar(horizons, depths, color=colors)
    chart_ax.set_ylabel("forecast depth (cm)")
    chart_ax.set_title("Forecast warning levels")
    chart_ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ymax = max(depths + [50.0]) * 1.2
    chart_ax.set_ylim(0, ymax)
    chart_ax.axhline(15, color="#2563eb", linestyle="--", linewidth=1.0)
    chart_ax.axhline(30, color="#ca8a04", linestyle="--", linewidth=1.0)
    chart_ax.axhline(50, color="#ea580c", linestyle="--", linewidth=1.0)
    for bar, depth, level in zip(bars, depths, levels):
        chart_ax.text(
            bar.get_x() + bar.get_width() / 2,
            depth + ymax * 0.02,
            f"{depth:.1f} cm\n{level}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#0f172a",
        )

    action_ax = fig.add_subplot(grid[2, :])
    action_ax.set_axis_off()
    action_ax.text(
        0.02,
        0.68,
        "Action suggestion",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
    )
    action_ax.text(
        0.02,
        0.34,
        textwrap.fill(action_short, width=125),
        ha="left",
        va="center",
        fontsize=10,
        color="#334155",
    )

    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def visualize_warning_summary(config_path: str | Path, project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).expanduser().resolve()
    config = load_warning_config(config_path)
    json_dir = resolve_project_path(root, config["output"]["json_dir"])
    figure_dir = resolve_project_path(root, config["output"]["figure_dir"])
    decision_path = json_dir / "warning_decision_result.json"
    output_path = figure_dir / "warning_summary.png"
    decision = load_json(decision_path)
    save_warning_summary(decision, output_path)
    print(f"[S8][summary] figure path: {output_path}")
    return {"warning_summary": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="S8 warning summary visualization.")
    parser.add_argument("--config", required=True, help="Path to configs/warning_config.yaml")
    parser.add_argument("--project_root", default=Path.cwd(), help="water_agent_system project root")
    args = parser.parse_args()
    visualize_warning_summary(args.config, args.project_root)


if __name__ == "__main__":
    main()
