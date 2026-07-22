# water_agent_system

## Project Overview

`water_agent_system` is a two-stage road-waterlogging perception research
prototype. A dry-road LiDAR survey builds the ground DEM; during rainfall,
Camera video, automatic temporal prompts, SAM 2 video propagation, shoreline
geometry, and the ground DEM estimate water level, depth, area, and volume.

The current release is a reproducible **synthetic shadow demo**. It does not
start live LiDAR, Camera, ROS nodes, Gazebo, or rosbag replay during the
competition demo, and it cannot generate authoritative warnings.

## Current Version

Current version tag:

- `v1.1-competition-synthetic-shadow-demo`

## Current Two-Stage Perception Pipeline

```text
dry LiDAR survey -> ground DEM
synthetic rainfall Camera video -> temporal evidence -> automatic SAM 2 prompt
-> SAM 2 video propagation -> outer shoreline -> ray/DEM intersection
-> water level -> depth/area/volume -> candidate quality gate
-> canonical shadow state -> read-only monitoring dashboard
```

Seed 303 final confirmation contains 12 sequences and 492 frames. The frozen
candidate gate passed 361 frames and rejected 131; all 361 passed frames were
within the 3 cm water-level-error research target, with no observed false pass
in that synthetic confirmation set. This result does not replace real-road
validation.

The single-Camera result distinguishes `global_scene_estimate` from
`camera_visible_estimate`. An unobservable secondary basin is not filled from
the DEM prior alone.

## Competition Synthetic Demo

The dedicated competition entry point only displays frozen simulation-road
Camera frames and C8/C9 prediction-side artifacts. It rejects Ground Truth,
manual-prompt, dormitory/cardboard, and non-simulation input paths.

```bash
cd ~/water_agent_ws/water_agent_system
bash scripts/run_phase2d_c10_competition_demo.sh
```

Open `http://localhost:8501`. The four fixed moderate-rain scenes show 5, 10,
20, and 40 cm simulation settings. The 5 cm case remains rejected, and the
40 cm case remains a partial Camera-visible estimate.

## Legacy Offline MVP Pipeline

The original S1-S8 pipeline remains available as a historical engineering
baseline and regression target. It is not the default competition perception
demo.

The reproducible offline demo runs this chain:

```text
S4 -> S5 -> S6 -> S7-A -> S7-B -> S7-C -> S8 -> Agent -> SQLite audit
```

The Agent stage order is:

```text
area_volume -> weather_correction -> deterministic_forecast ->
case_retrieval -> physical_constraint -> warning_report
```

## Quick Start

```bash
cd ~/water_agent_ws/water_agent_system
bash scripts/setup_env.sh
source .venv/bin/activate
python3 scripts/check_project_health.py
bash scripts/run_full_offline_demo.sh
```

Key outputs:

- `outputs/json/health_check_result.json`
- `outputs/json/agent_run_summary.json`
- `outputs/json/final_forecast_result.json`
- `outputs/json/warning_decision_result.json`
- `outputs/reports/warning_report.md`
- `outputs/figures/warning_summary.png`

## Streamlit Dashboard

The project includes a Streamlit dashboard（可视化看板） for teacher
review, classmate collaboration, and offline project demos. It displays
pipeline diagrams, result figures, JSON metrics, warning reports, quality
gate（质量门控） conclusions, S4-real diagnostics, and Agent audit
summaries.

Run:

```bash
cd ~/water_agent_ws/water_agent_system
streamlit run dashboard/app.py
```

For competition presentation, prefer the dedicated simulation-only entry point
shown above; it does not include the historical dormitory comparison pages.

If the browser does not open automatically, open the local URL printed by
Streamlit, usually `http://localhost:8501`. The dashboard only reads
offline output files and does not start LiDAR, camera, ROS nodes, or
rosbag replay.

## Important Notes

- The current competition result is synthetic and non-authoritative.
- Automatic temporal prompts and SAM 2 video propagation are implemented;
  the competition path does not require per-frame manual clicks.
- Dynamic rainfall simulation is an offline research and regression tool, not
  a future production runtime dependency.
- The 5 cm shallow-water condition remains the main visual limitation.
- Single-Camera coverage cannot safely recover unobservable basins; multi-Camera
  coverage is a future direction.
- Formal S5-S8 warning activation remains disabled.
- The current demo does not start real-time devices.
- The current configured depth is an MVP simulation.
- The current weather input is offline mock weather.
- The current depth history is offline mock depth history.
- The current case library is an offline mock case library.
- The current physical constraint is a simplified MVP water-balance check.
- Future work can replace these pieces with real point-cloud water-surface
  DEM, a real meteorological API, rosbag replay, and real-time processing.

## Git Tags

- `v0.1-offline-agent-mvp`
- `v0.2-case-retrieval-mvp`
- `v0.3-physical-constraint-mvp`
- `v0.4-full-s7-agent-integration`
- `v0.5-reproducible-offline-demo`
- `v1.0-project-demo-ready`
- `v1.1-competition-synthetic-shadow-demo`

## Real LiDAR Surface DEM Depth Inversion

The original S4 `configured_depth` path is kept for MVP pipeline
verification. The new S4-real path reads offline LiDAR point clouds from
existing rosbags and builds a current surface DEM. It then computes:

```text
surface_depth = current_surface_dem - ground_dem_interpolated
```

This mode does not start real-time devices. It only reads offline rosbags
with `rosbag2_py`. The result does not use `configured_depth`, but its
accuracy still depends on calibration quality, ROI mapping, point density,
and LiDAR returns from the water/surface target.

Current dormitory validation is diagnostic only. The 13cm dormitory scene
is clearly overestimated, while the 39cm scene is much closer to the known
manual depth. Low valid-cell coverage can affect accuracy, and a playground
open-scene dataset should be recollected as the more formal S4-real
validation set.

Run the 13cm scene:

```bash
source /opt/ros/humble/setup.bash
cd ~/water_agent_ws/water_agent_system

python3 src/dem/build_surface_dem_from_rosbag.py --config configs/surface_dem_config.yaml --case water_sim_13cm_001
python3 src/hydrology/invert_surface_depth.py --config configs/surface_dem_config.yaml --case water_sim_13cm_001
python3 src/evaluation/evaluate_surface_depth_accuracy.py --config configs/surface_dem_config.yaml --case water_sim_13cm_001
```

Run the 39cm scene:

```bash
python3 src/dem/build_surface_dem_from_rosbag.py --config configs/surface_dem_config.yaml --case water_sim_39cm_001
python3 src/hydrology/invert_surface_depth.py --config configs/surface_dem_config.yaml --case water_sim_39cm_001
python3 src/evaluation/evaluate_surface_depth_accuracy.py --config configs/surface_dem_config.yaml --case water_sim_39cm_001
```

Run quality diagnosis for both cases:

```bash
python3 src/evaluation/diagnose_surface_depth_quality.py --config configs/surface_dem_config.yaml
```

Run the S4-real quality gate for a case:

```bash
python3 run_offline_pipeline.py \
  --stage surface_depth_quality_gate \
  --case playground_pit_water_sim_6cm_001
```

The quality gate labels each S4-real result as `pass`, `warning`, or
`reject` before it can be considered for downstream S5-S8 warning use.
Rejected S4-real results are saved as diagnostic artifacts only and
must not enter the formal warning chain.

## S4-real-B Boundary Waterline Inversion

S4-real-B adds a boundary-based waterline method for shallow-water or
unstable water-surface return scenes. Instead of directly trusting LiDAR
surface returns, it:

1. uses a water-region mask to define the flooded area,
2. extracts boundary cells from that mask,
3. reads boundary ground elevations from the ground DEM,
4. estimates a water level with a trimmed median,
5. computes depth as `max(0, water_level - ground_dem)` inside the mask,
6. reports depth, area, volume, and a boundary-specific quality status.

This method can be useful when LiDAR cannot reliably detect the water
surface, but it depends on mask boundary quality and DEM quality. The
playground scene now requires its own DEM-space mask and refuses to use
the old dormitory mask as a fallback.

### S3-playground DEM-space Mask

The playground pit scene requires a DEM-space mask whose shape exactly
matches `data/dem/playground_pit/ground_dem_interpolated.npy`. The older
dormitory `data/fusion/water_region_mask.npy` has a different shape and
must not be used for playground S4-real-B.

Create the playground DEM-space mask:

```bash
cd ~/water_agent_ws/water_agent_system
python3 run_offline_pipeline.py \
  --stage create_dem_water_mask \
  --config configs/playground_pit_dem_mask_config.yaml
```

The polygon points are row/column DEM grid coordinates, not camera
pixels. Review the generated overlay figures and adjust
`configs/playground_pit_dem_mask_config.yaml` if the mask does not cover
only the pit water region.

Diagnose the current playground mask before using S4-real-B as evidence:

```bash
python3 src/masks/diagnose_dem_space_mask.py \
  --config configs/playground_pit_dem_mask_config.yaml
```

The diagnosis reports mask area, boundary height spread, boundary
outliers, and point-count statistics. It also writes overlay figures for
manual refinement. If needed, edit `refined_polygon_points_rc` and
regenerate the mask with:

```bash
python3 src/masks/create_dem_space_water_mask.py \
  --config configs/playground_pit_dem_mask_config.yaml \
  --use-refined
```

Do not refine the polygon solely to force agreement with `known_depth_cm`;
the mask boundary should match the actual pit water region in the DEM
and point-count overlays.

Run S4-real-B for the playground pit 6cm case:

```bash
source /opt/ros/humble/setup.bash
cd ~/water_agent_ws/water_agent_system

python3 run_offline_pipeline.py \
  --stage boundary_waterline_depth \
  --case playground_pit_water_sim_6cm_001
```

Outputs:

- `data/hydrology/<case>/boundary_waterline_depth_map.npy`
- `data/hydrology/<case>/boundary_waterline_depth_valid_mask.npy`
- `outputs/json/boundary_waterline_depth_result_<case>.json`
- `outputs/figures/boundary_waterline_depth_heatmap_<case>.png`
- `outputs/reports/boundary_waterline_depth_report.md`

Run the playground pit 6cm scene:

```bash
source /opt/ros/humble/setup.bash
cd ~/water_agent_ws/water_agent_system

python3 run_offline_pipeline.py --stage build_ground_dem --case playground_pit_dry_baseline_001
python3 run_offline_pipeline.py --stage build_surface_dem --case playground_pit_water_sim_6cm_001
python3 run_offline_pipeline.py --stage surface_depth --case playground_pit_water_sim_6cm_001
python3 run_offline_pipeline.py --stage surface_depth_eval --case playground_pit_water_sim_6cm_001

python3 src/evaluation/diagnose_surface_depth_quality.py \
  --config configs/surface_dem_config.yaml \
  --cases playground_pit_water_sim_6cm_001
```

The playground pit scene uses `known_depth_cm = 6.0`. Its water case must
use the scene-specific ground DEM built from
`playground_pit_dry_baseline_001`; it must not use the dormitory ground DEM.
The current playground 6cm result is clearly overestimated and has low
valid-depth coverage, so the quality gate rejects it for downstream
warning use.

Pipeline entrypoint equivalents:

```bash
python3 run_offline_pipeline.py --stage build_surface_dem --config configs/surface_dem_config.yaml --case water_sim_13cm_001
python3 run_offline_pipeline.py --stage surface_depth --config configs/surface_dem_config.yaml --case water_sim_13cm_001
python3 run_offline_pipeline.py --stage surface_depth_eval --config configs/surface_dem_config.yaml --case water_sim_13cm_001
```

`water_agent_system` 是面向城市道路积水感知与预警的离线工程化原型。

当前已实现说明书 **S2：无水 DEM 构建模块** 的两个离线版本：

- **S2-A `dem_baseline`**：初版场景高度栅格图，包含地面、墙面、门、柜子、远处立面等可观测高度。
- **S2-B `ground_dem`**：无水地面高程基准，通过地面高度过滤和低分位栅格统计得到，后续 S4 水深反演应使用这个结果。
- **S3 `manual_water_mask`**：人工 polygon mask，作为图像分割的最小可运行实现。
- **S4 `water_depth_map`**：区域级手动 mask-to-DEM 映射和 configured-depth 水深图，验证水深反演管线结构。

## 当前阶段

- 输入数据：`~/water_agent_data/rosbags/dry_baseline_001`
- 输入 topic：`/cx/lslidar_point_cloud`
- 处理方式：离线读取 ROS2 rosbag，不依赖实时 LiDAR 或摄像头节点
- 输出目标：构建无水地面 DEM，作为后续 S4 水深反演的空间基准

当前不训练深度学习模型、不做水深反演、不做趋势预测。

## 目录结构

```text
water_agent_system/
├── configs/
├── data/
│   ├── dem/
│   ├── masks/
│   ├── cases/
│   └── audit_logs/
├── src/
│   ├── sensors/
│   ├── dem/
│   ├── vision/
│   ├── fusion/
│   ├── hydrology/
│   ├── inference/
│   ├── warning/
│   └── agent/
├── outputs/
│   ├── figures/
│   ├── json/
│   └── reports/
├── tests/
├── run_offline_pipeline.py
└── README.md
```

## 运行

S2-A 初版场景高度栅格图：

```bash
cd ~/water_agent_ws/water_agent_system
python3 run_offline_pipeline.py \
  --stage dem \
  --dry_bag ~/water_agent_data/rosbags/dry_baseline_001 \
  --config configs/system_config.yaml
```

S2-B 无水地面高程基准：

```bash
cd ~/water_agent_ws/water_agent_system
python3 run_offline_pipeline.py \
  --stage ground_dem \
  --dry_bag ~/water_agent_data/rosbags/dry_baseline_001 \
  --config configs/system_config.yaml
```

S3 离线抽取相机帧：

```bash
cd ~/water_agent_ws/water_agent_system
python3 run_offline_pipeline.py \
  --stage extract_camera \
  --bag ~/water_agent_data/rosbags/water_sim_39cm_001 \
  --config configs/system_config.yaml \
  --output outputs/figures/camera_water_sim_39cm.png
```

S3 人工 polygon mask：

```bash
cd ~/water_agent_ws/water_agent_system
python3 run_offline_pipeline.py \
  --stage manual_mask \
  --config configs/manual_mask_config.yaml
```

S4 区域级 mask-to-DEM 映射：

```bash
cd ~/water_agent_ws/water_agent_system
python3 run_offline_pipeline.py \
  --stage mask_to_dem \
  --config configs/roi_mapping.yaml
```

S4 configured-depth 水深图：

```bash
cd ~/water_agent_ws/water_agent_system
python3 run_offline_pipeline.py \
  --stage water_depth \
  --config configs/roi_mapping.yaml
```

如果当前终端尚未加载 ROS2 环境，请先执行：

```bash
source /opt/ros/humble/setup.bash
source ~/water_agent_ws/install/setup.bash
```

## S2-A 输出

`dem_baseline` 是初版场景高度栅格图，主要用于检查点云覆盖和空间范围，不应作为最终水深反演基准。

- `data/dem/dem_baseline.npy`
- `data/dem/dem_valid_mask.npy`
- `data/dem/dem_point_count.npy`
- `data/dem/dem_metadata.json`
- `outputs/figures/dem_baseline_heatmap.png`

## S2-B 输出

`ground_dem` 是说明书 S2 所需的无水地面高程基准。后续 S4 水深反演应使用 `ground_dem`，而不是 `dem_baseline`。

- `data/dem/ground_dem.npy`
- `data/dem/ground_dem_valid_mask.npy`
- `data/dem/ground_dem_point_count.npy`
- `data/dem/ground_dem_interpolated.npy`
- `data/dem/ground_dem_metadata.json`
- `outputs/figures/ground_dem_heatmap.png`
- `outputs/figures/ground_dem_interpolated_heatmap.png`
- `outputs/figures/ground_dem_point_count.png`

终端会打印 DEM grid size、DEM shape、valid cell count、z_min / z_max / z_median 和输出文件路径。

## S3 输出

当前 S3 使用人工 polygon mask 作为图像分割最小实现。后续可替换为 YOLO-seg、SAM、Mask2Former 等自动分割模块。

`manual_water_mask` 是后续 S4 mask-to-DEM 映射和水深反演的输入。

- `outputs/figures/camera_water_sim_39cm.png`
- `data/masks/manual_water_mask.png`
- `data/masks/manual_water_mask.npy`
- `data/masks/manual_water_mask_metadata.json`
- `outputs/figures/manual_water_mask_overlay.png`

如果 polygon 覆盖区域不合适，请人工调整 `configs/manual_mask_config.yaml` 中的 `polygon_points`。

## S4 输出

当前 S4 使用区域级手动映射，不是像素级外参标定。`water_surface.mode = configured_depth` 只用于验证说明书 S4 的管线结构，结果会标注为 `configured_mvp_simulation`，不代表最终真实水深测量。

后续可替换为真实点云水面高程，或“图像边界 + DEM 反推水面高度”。

- `data/fusion/water_region_mask.npy`
- `data/fusion/mask_to_dem_mapping.json`
- `outputs/figures/water_region_on_dem.png`
- `data/hydrology/water_depth_map.npy`
- `data/hydrology/water_depth_valid_mask.npy`
- `outputs/figures/water_depth_heatmap.png`
- `outputs/json/water_depth_result.json`

## S5 Output

S5 calculates water area and water volume from the S4 outputs:
`data/hydrology/water_depth_map.npy` and
`data/hydrology/water_depth_valid_mask.npy`.

The current S5 values inherit the S4 `configured_mvp_simulation`
depth method. They are used only to validate the S4-S5 engineering
pipeline and are not final real water-depth measurements.

After S4 is replaced by a real LiDAR water-surface elevation source,
the S5 area and volume calculation can remain mostly unchanged; only
the S4 depth source needs to be replaced.

```bash
cd ~/water_agent_ws/water_agent_system
python3 run_offline_pipeline.py \
  --stage area_volume \
  --config configs/roi_mapping.yaml
```

- `data/hydrology/water_area_volume_result.json`
- `outputs/json/water_area_volume_result.json`
- `outputs/figures/water_area_volume_summary.png`

## S6 Output

S6 currently uses offline mock rainfall data from
`configs/weather_config.yaml`. It does not call a real meteorological
API in this MVP stage; it only validates that the weather correction
factor can enter the S7 engineering pipeline.

Correction factor rules:

- no rain: `0.7`
- `0-15 mm/h`: `1.0`
- `15-30 mm/h`: `1.3`
- `>=30 mm/h`: `1.8`

Future work can replace `offline_mock` with a real weather API while
keeping the same JSON fields for S7.

```bash
cd ~/water_agent_ws/water_agent_system
python3 run_offline_pipeline.py \
  --stage weather_correction \
  --config configs/weather_config.yaml
```

- `data/meteorology/weather_correction_result.json`
- `outputs/json/weather_correction_result.json`
- `outputs/figures/weather_correction_summary.png`

## S7-A Output

S7-A implements the first layer of the three-layer hybrid reasoning
module: a deterministic rule-engine MVP. It reads S5 area/volume/depth
statistics and the S6 weather correction factor.

The current version uses `offline_mock_depth_history` to simulate
1/5/10 minute depth-change rates, then calculates `k_base`,
`k_forecast`, and forecast depths for the next 5/15/30/60 minutes.

This is not a final real short-term forecast. It uses
`configured_mvp_simulation`, `offline_mock_weather`, and
`offline_mock_depth_history` only to validate the S7 pipeline. Later
work should replace mock history with real sequential S4-S5 outputs.
S7-B will add case-retrieval correction, and S7-C will add physical
water-balance constraints.

```bash
cd ~/water_agent_ws/water_agent_system
python3 run_offline_pipeline.py \
  --stage deterministic_forecast \
  --config configs/prediction_config.yaml
```

- `data/reasoning/deterministic_forecast_result.json`
- `outputs/json/deterministic_forecast_result.json`
- `outputs/figures/deterministic_forecast_curve.png`

## S7-B Output

S7-B implements a case retrieval correction MVP. It reads the S5
area/volume result, S6 weather correction result, and S7-A deterministic
forecast result, then compares the current event with
`offline_mock_case_library` cases using weighted Euclidean distance.

For the top-k retrieved mock cases, S7-B takes the median historical
forecast bias for the 5/15/30/60 minute horizons and applies that bias
to the deterministic forecast. This validates the case-retrieval layer
only; it is not final real historical-case correction.

```bash
cd ~/water_agent_ws/water_agent_system
python3 run_offline_pipeline.py \
  --stage case_retrieval \
  --config configs/case_retrieval_config.yaml
```

- `data/reasoning/case_retrieval_result.json`
- `outputs/json/case_retrieval_result.json`
- `data/reasoning/corrected_forecast_result.json`
- `outputs/json/corrected_forecast_result.json`
- `outputs/figures/case_retrieval_correction.png`

## S7-C Output

S7-C implements a simplified physical constraint check MVP. It uses
`simplified_water_balance_mvp` to compare S7-B corrected forecast
volume against a simple expected volume:

`current volume + rainfall input - drainage output - infiltration loss`

The current version uses a linear volume-depth proxy based on S5:
`volume_per_cm = current_volume_m3 / current_mean_depth_cm`.
If the forecast volume deviates beyond the configured tolerance, S7-C
applies a limited callback toward the expected depth and recalculates
the warning level.

This is not a full hydrodynamic model and not SWMM. It only validates
the physical-constraint layer for the MVP. Future versions can use real
catchment area, real drainage capacity, and a hydrodynamic/SWMM model.

```bash
cd ~/water_agent_ws/water_agent_system
python3 run_offline_pipeline.py \
  --stage physical_constraint \
  --config configs/physical_constraint_config.yaml
```

- `data/reasoning/physical_constraint_result.json`
- `outputs/json/physical_constraint_result.json`
- `data/reasoning/final_forecast_result.json`
- `outputs/json/final_forecast_result.json`
- `outputs/figures/physical_constraint_summary.png`

## S8 Output

S8 generates graded warning decisions and action suggestions from the
S7-A forecast result.

Warning rules:

- `<15 cm`: `none`
- `15-30 cm`: `blue`
- `30-50 cm`: `yellow`
- `>=50 cm`: `orange`

Current outputs:

- `data/warnings/warning_decision_result.json`
- `outputs/json/warning_decision_result.json`
- `outputs/reports/warning_report.md`
- `data/audit_logs/warning_audit_log.jsonl`
- `outputs/figures/warning_summary.png`

S8 is still MVP simulation and is not final real emergency dispatch
advice. A later Agent MVP will orchestrate S4-S8 and record unified
audit logs.

```bash
cd ~/water_agent_ws/water_agent_system
python3 run_offline_pipeline.py \
  --stage warning_report \
  --config configs/warning_config.yaml
```

## Agent MVP And SQLite Audit DB

The Agent MVP is an `offline_pipeline_agent`. It orchestrates the
existing S4-S8 offline modules and does not start real-time devices or
modify drivers.

Each run is written into a SQLite audit database at:

- `data/db/water_agent_audit.db`

The database is for auditing and project demonstration. It stores run
metadata, stage status, key indicators, and artifact paths only. It
does not store rosbag files, large `.npy` arrays, or image/report
contents.

```bash
cd ~/water_agent_ws/water_agent_system
python3 src/agent/pipeline_agent.py \
  --config configs/agent_config.yaml

python3 run_offline_pipeline.py \
  --stage agent_pipeline \
  --config configs/agent_config.yaml

python3 src/database/show_audit_db.py \
  --db data/db/water_agent_audit.db
```

Generated Agent outputs:

- `data/agent/agent_run_summary.json`
- `outputs/json/agent_run_summary.json`
- `data/db/water_agent_audit.db`

Future versions can split this MVP into a perception Agent, modeling
Agent, meteorology Agent, reasoning Agent, warning Agent, and audit
Agent.

## Integrated S7 Hybrid Reasoning And S8 Warning

The current offline pipeline integrates the full S7 three-layer hybrid
reasoning chain before S8 warning generation:

1. **S7-A deterministic forecast** reads S5 area/volume results and S6
   weather correction, then produces deterministic 5/15/30/60 minute
   forecast depths.
2. **S7-B case retrieval correction** reads the S7-A forecast and an
   `offline_mock_case_library`, then applies median historical bias from
   top-k mock similar cases.
3. **S7-C physical constraint check** reads the S7-B corrected forecast
   and applies `simplified_water_balance_mvp` constraints to produce
   `final_forecast_result.json`.
4. **S8 warning decision** now prefers
   `outputs/json/final_forecast_result.json` when it exists. If it is
   missing, S8 falls back to
   `outputs/json/deterministic_forecast_result.json` and records a
   fallback warning in `warning_decision_result.json`.
5. **Agent MVP** now schedules the offline chain:
   `area_volume -> weather_correction -> deterministic_forecast ->
   case_retrieval -> physical_constraint -> warning_report`.

This remains an MVP simulation. The current chain uses
`configured_mvp_simulation`, `offline_mock_weather`,
`offline_mock_depth_history`, `offline_mock_case_library`, and
`simplified_water_balance_mvp`. It is not a real-time emergency dispatch
system and not a final hydrodynamic/SWMM forecast.

## 老师汇报材料

本项目已生成老师汇报用材料，默认输出位置优先使用 Windows 共享桌面：

- `/mnt/hgfs/WinDesktop/water_agent_report`
- `/mnt/hgfs/Desktop/water_agent_report`

如果上述共享目录不存在，则输出到项目目录：

- `outputs/final_report_for_windows`

材料包括：

- `项目汇报报告.md`
- `老师汇报讲稿.md`
- `演示操作说明.md`
- `项目当前状态总结.md`

如果系统支持 `pandoc` 或 `python-docx`，同时会生成对应 `.docx` 文件。

Streamlit dashboard（可视化看板）运行方式：

```bash
cd ~/water_agent_ws/water_agent_system
streamlit run dashboard/app.py
```

dashboard 只读取离线输出文件，不启动实时 LiDAR、摄像头、ROS 节点或 rosbag replay。

