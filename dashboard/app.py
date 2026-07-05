#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit dashboard for water_agent_system offline results."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils import (  # noqa: E402
    file_summary,
    format_number,
    json_table,
    metric_grid,
    metric_value,
    project_root,
    read_json,
    section_note,
    show_image,
    show_json,
    show_markdown,
    status_badge,
)


TITLE = "基于路侧多模态感知的低洼道路积水检测与短临预警系统"

PAGES = [
    "首页：项目概览",
    "S4-MVP：configured_depth 管线",
    "S4-real-A：surface DEM 直接反演",
    "S4-real：质量门控",
    "S4-real-B：边界水位线反演",
    "DEM-space mask（数字高程掩膜）诊断",
    "S7/S8：短临预警链路",
    "Agent（智能体）与审计",
]


def page_home() -> None:
    st.title(TITLE)
    st.caption("dashboard（可视化看板） | offline MVP（离线最小可运行系统）")
    st.write(
        "雨季低洼路段容易形成积水，设备部署在路侧、路灯杆或监控杆高度，"
        "通过 LiDAR（激光雷达）和摄像头监测积水区域、水深、面积、体积和未来风险，"
        "提前预警，减少车辆涉水和财产损失。"
    )

    st.subheader("系统流程")
    stages = [
        ("S1", "多模态采集", "LiDAR + 摄像头 rosbag（ROS 数据包）"),
        ("S2", "ground DEM", "无水地面 DEM（数字高程模型）基准"),
        ("S3", "积水 mask", "图像 mask（掩膜）/ DEM-space mask（DEM 栅格掩膜）"),
        ("S4", "水深反演", "configured_depth、surface DEM、boundary waterline"),
        ("S5", "面积体积", "积水面积、体积和水深统计"),
        ("S6", "气象修正", "offline mock weather（离线模拟气象）"),
        ("S7", "三层推理", "规则引擎 + 案例检索 + 物理约束"),
        ("S8", "分级预警与审计日志", "预警报告、行动建议、SQLite 审计"),
    ]
    for code, name, desc in stages:
        st.markdown(f"**{code} {name}**：{desc}")

    st.subheader("演示边界")
    st.warning(
        "本 dashboard 只展示离线结果，不启动实时 LiDAR、摄像头、ROS 节点或 rosbag replay。"
        "S4-real experimental（实验阶段）结果需要经过 quality gate（质量门控）后才能进入正式预警链路。"
    )
    st.caption(f"项目根目录：`{project_root()}`")


def page_s4_mvp() -> None:
    st.title("S4-MVP：configured_depth 管线验证")
    section_note(
        "`configured_depth` 是 MVP 管线验证，不是真实最终水深测量。"
        "它用于证明 mask → DEM → 水深图 → 面积体积的工程链路能跑通。"
    )
    show_image("outputs/figures/water_depth_heatmap.png", "S4 MVP water_depth_heatmap")

    depth = read_json("outputs/json/water_depth_result.json")
    area = read_json("outputs/json/water_area_volume_result.json")
    metric_grid(
        [
            ("configured depth", format_number(metric_value(depth, "configured_depth_cm"), 2, " cm"), "配置水深"),
            ("max depth", format_number(metric_value(depth, "max_depth_cm"), 2, " cm"), "最大水深"),
            ("water area", format_number(metric_value(area, "water_area_m2"), 2, " m²"), "积水面积"),
            ("water volume", format_number(metric_value(area, "water_volume_m3"), 4, " m³"), "积水体积"),
        ]
    )
    show_json("outputs/json/water_depth_result.json", "water_depth_result.json")
    show_json("outputs/json/water_area_volume_result.json", "water_area_volume_result.json")


def page_s4_real_a() -> None:
    st.title("S4-real-A：surface DEM 直接水深反演")
    section_note(
        "surface DEM（当前表面数字高程模型）直接反演读取离线 LiDAR 点云，"
        "计算 current_surface_dem - ground_dem。当前作为 S4-real experimental（实验阶段）结果展示。"
    )

    cols = st.columns(2)
    with cols[0]:
        st.subheader("宿舍 13cm")
        show_image(
            "outputs/figures/surface_water_depth_heatmap_water_sim_13cm_001.png",
            "surface_water_depth_heatmap_water_sim_13cm_001",
        )
        case13 = read_json("outputs/json/surface_depth_accuracy_water_sim_13cm_001.json")
        json_table(
            case13 if isinstance(case13, dict) else None,
            ["known_depth_cm", "mean_depth_cm", "median_depth_cm", "max_depth_cm", "mean_error_cm", "valid_depth_ratio_in_water_region"],
        )
    with cols[1]:
        st.subheader("宿舍 39cm")
        show_image(
            "outputs/figures/surface_water_depth_heatmap_water_sim_39cm_001.png",
            "surface_water_depth_heatmap_water_sim_39cm_001",
        )
        case39 = read_json("outputs/json/surface_depth_accuracy_water_sim_39cm_001.json")
        json_table(
            case39 if isinstance(case39, dict) else None,
            ["known_depth_cm", "mean_depth_cm", "median_depth_cm", "max_depth_cm", "mean_error_cm", "valid_depth_ratio_in_water_region"],
        )

    st.subheader("结论")
    st.markdown(
        "- 39cm 可控场景接近人工水深。\n"
        "- 13cm 场景明显高估。\n"
        "- 有效覆盖率偏低，结果需要谨慎解释。\n"
        "- 当前作为 S4-real experimental（实验阶段）结果。"
    )
    show_json("outputs/json/surface_depth_accuracy_water_sim_13cm_001.json", "13cm accuracy JSON")
    show_json("outputs/json/surface_depth_accuracy_water_sim_39cm_001.json", "39cm accuracy JSON")
    show_json("outputs/json/surface_depth_quality_diagnosis.json", "surface_depth_quality_diagnosis.json")


def page_quality_gate() -> None:
    st.title("S4-real quality gate（质量门控）")
    gate = read_json("outputs/json/surface_depth_quality_gate_playground_pit_water_sim_6cm_001.json")
    if isinstance(gate, dict):
        status_badge("quality_status", gate.get("quality_status"))
        status_badge("can_enter_s5_s8_warning_chain", gate.get("can_enter_s5_s8_warning_chain"))
        metric_grid(
            [
                ("valid ratio", format_number(gate.get("valid_depth_ratio_in_water_region"), 4), "有效覆盖率"),
                ("mean error", format_number(gate.get("mean_error_cm"), 2, " cm"), "平均误差"),
                ("max depth", format_number(gate.get("max_depth_cm"), 2, " cm"), "最大水深"),
                ("known depth", format_number(gate.get("known_depth_cm"), 2, " cm"), "人工已知水深"),
            ]
        )
        st.subheader("reject（拒绝进入后续链路）原因")
        for reason in gate.get("reject_reasons", []):
            st.error(reason)
    else:
        st.warning("缺少 quality gate（质量门控）JSON。")

    st.markdown(
        "重点结论：`playground_pit_water_sim_6cm_001` 由于低覆盖率、高均值误差、"
        "极端最大水深异常，以及 `high_error_warning` 与 `coverage_warning` 同时触发，"
        "被判定为 reject（拒绝进入后续链路）。"
    )
    show_json("outputs/json/surface_depth_quality_gate_playground_pit_water_sim_6cm_001.json")
    show_markdown("outputs/reports/surface_depth_quality_gate_report.md", "质量门控报告")


def page_boundary_waterline() -> None:
    st.title("S4-real-B：边界水位线反演")
    section_note(
        "该方法是 S4-real-B fallback（备用）路线。当 LiDAR 难以直接检测浅水水面高程时，"
        "尝试使用积水边界和 ground DEM（地面数字高程模型）估计水面高度。"
        "当前 playground_pit 6cm 因 mask（掩膜）边界高程离散过大，被判定为 reject。"
    )
    show_image(
        "outputs/figures/boundary_waterline_depth_heatmap_playground_pit_water_sim_6cm_001.png",
        "boundary_waterline_depth_heatmap_playground_pit_water_sim_6cm_001",
    )
    result = read_json("outputs/json/boundary_waterline_depth_result_playground_pit_water_sim_6cm_001.json")
    if isinstance(result, dict):
        status_badge("boundary_quality_status", result.get("boundary_quality_status"))
        metric_grid(
            [
                ("water level", format_number(result.get("estimated_water_level_m"), 4, " m"), "估计水位线"),
                ("boundary std", format_number(result.get("boundary_height_std_cm"), 2, " cm"), "边界高程标准差"),
                ("mean depth", format_number(result.get("mean_depth_cm"), 2, " cm"), "平均水深"),
                ("area", format_number(result.get("area_m2"), 2, " m²"), "积水面积"),
            ]
        )
    show_json("outputs/json/boundary_waterline_depth_result_playground_pit_water_sim_6cm_001.json")
    show_markdown("outputs/reports/boundary_waterline_depth_report.md", "S4-real-B 报告")


def page_mask_diagnosis() -> None:
    st.title("DEM-space mask（数字高程栅格掩膜）诊断")
    section_note(
        "当前操场坑洼 mask 边界高程离散大，point_count（点云数量）median 为 0，"
        "说明操场坑洼数据采集条件和 mask 边界质量不足，需要人工精修 `refined_polygon_points_rc`。"
    )
    cols = st.columns(3)
    with cols[0]:
        show_image("outputs/figures/playground_pit_mask_diagnosis_on_dem.png", "mask on ground DEM")
    with cols[1]:
        show_image("outputs/figures/playground_pit_mask_boundary_height_outliers.png", "boundary height outliers")
    with cols[2]:
        show_image("outputs/figures/playground_pit_mask_on_point_count.png", "mask on point count")

    diag = read_json("outputs/json/playground_pit_mask_diagnosis.json")
    if isinstance(diag, dict):
        metric_grid(
            [
                ("mask area", format_number(diag.get("mask_area_m2"), 2, " m²"), "mask 面积"),
                ("boundary std", format_number(diag.get("boundary_height_std_cm"), 2, " cm"), "边界高程标准差"),
                ("valid boundary ratio", format_number(diag.get("boundary_valid_ratio"), 4), "有效边界比例"),
                ("mask cells", diag.get("mask_cell_count", "N/A"), "mask 栅格数量"),
            ]
        )
        st.subheader("边界异常点统计")
        st.json(diag.get("boundary_outlier_counts", {}), expanded=True)
        st.subheader("point_count（点云数量）统计")
        st.json(
            {
                "point_count_on_mask_stats": diag.get("point_count_on_mask_stats", {}),
                "point_count_on_boundary_stats": diag.get("point_count_on_boundary_stats", {}),
            },
            expanded=False,
        )
    show_json("outputs/json/playground_pit_mask_diagnosis.json")
    show_markdown("outputs/reports/playground_pit_mask_diagnosis.md", "mask 诊断报告")


def page_s7_s8() -> None:
    st.title("S7/S8：短临预警链路")
    section_note(
        "这部分展示完整短临预警链路。如果 S4-real quality gate（质量门控）为 reject，"
        "真实 S4-real 结果不应直接进入正式预警链路；这里只展示已有离线 MVP 预警链路。"
    )
    cols = st.columns(3)
    with cols[0]:
        show_image("outputs/figures/deterministic_forecast_curve.png", "deterministic_forecast_curve")
    with cols[1]:
        show_image("outputs/figures/physical_constraint_summary.png", "physical_constraint_summary")
    with cols[2]:
        show_image("outputs/figures/warning_summary.png", "warning_summary")

    forecast = read_json("outputs/json/final_forecast_result.json")
    warning = read_json("outputs/json/warning_decision_result.json")
    if isinstance(warning, dict):
        status_badge("overall_warning_level", warning.get("overall_warning_level"))
        st.markdown(f"**action suggestion（行动建议）**：{warning.get('action_suggestion', 'N/A')}")
    if isinstance(forecast, dict):
        json_table(
            forecast,
            ["forecast_source", "overall_warning_level", "current_mean_depth_cm", "k_forecast_cm_per_min"],
        )
    show_json("outputs/json/final_forecast_result.json", "final_forecast_result.json")
    show_json("outputs/json/warning_decision_result.json", "warning_decision_result.json")
    show_markdown("outputs/reports/warning_report.md", "warning_report.md")


def page_agent_audit() -> None:
    st.title("Agent（智能体）与 SQLite 审计")
    st.markdown(
        "Agent stage 顺序：`area_volume -> weather_correction -> deterministic_forecast -> "
        "case_retrieval -> physical_constraint -> warning_report`"
    )
    summary = read_json("outputs/json/agent_run_summary.json")
    if isinstance(summary, dict):
        metric_grid(
            [
                ("run status", summary.get("status", "N/A"), "Agent 运行状态"),
                ("overall warning", summary.get("overall_warning_level", "N/A"), "总体预警等级"),
                ("water area", format_number(summary.get("water_area_m2"), 2, " m²"), "积水面积"),
                ("water volume", format_number(summary.get("water_volume_m3"), 4, " m³"), "积水体积"),
            ]
        )
    show_json("outputs/json/agent_run_summary.json", "agent_run_summary.json")
    st.subheader("SQLite 审计数据库")
    file_summary(["data/db/water_agent_audit.db"])
    st.caption("这里只显示数据库路径和存在性，不强制读取 SQLite 文件。")


def main() -> None:
    st.set_page_config(
        page_title="water_agent_system dashboard",
        page_icon="W",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    with st.sidebar:
        st.title("water_agent_system")
        page = st.radio("页面", PAGES)
        st.caption("offline dashboard（离线可视化看板）")

    page_map = {
        PAGES[0]: page_home,
        PAGES[1]: page_s4_mvp,
        PAGES[2]: page_s4_real_a,
        PAGES[3]: page_quality_gate,
        PAGES[4]: page_boundary_waterline,
        PAGES[5]: page_mask_diagnosis,
        PAGES[6]: page_s7_s8,
        PAGES[7]: page_agent_audit,
    }
    page_map[page]()


if __name__ == "__main__":
    main()
