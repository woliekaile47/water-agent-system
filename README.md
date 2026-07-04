# water_agent_system

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
