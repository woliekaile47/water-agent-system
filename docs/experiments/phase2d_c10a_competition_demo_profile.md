# Phase 2D-C-10A：比赛离线演示主链路与素材边界

## 目的

比赛离线演示固定展示项目生成的仿真道路、动态降雨和积水结果，不再把宿舍纸箱模拟积水、临时 `water_test` 图片或人工提示盲测素材作为默认演示输入。

本阶段不修改 C6 自动提示、C7 SAM 2 视频传播、C8 候选质量门控或 C9 旁路集成算法，只增加展示侧的固定配置、来源校验和 Dashboard 页面。

## 默认演示数据

固定使用 Seed 303、锚点帧 149 的四个 moderate rain 场景：

- `sim_water_5cm_001`；
- `sim_water_10cm_001`；
- `sim_water_20cm_001`；
- `sim_water_40cm_001`。

四个场景均来自 `data/simulation_dynamic/`，预测 mask 来自冻结的 C8 SAM 2 视频传播输出，水位、水深、面积和体积来自冻结的 C8 DEM 几何输出。5 cm 场景即使被质量门控拒绝也照实展示，不能为了演示效果更换样本或隐藏失败。

`nominal_depth_cm_display_only` 只用于标注仿真场景设置，不进入预测计算，也不用于改变候选选择或质量门控。

## 来源安全规则

构建器只允许读取：

- `data/simulation_dynamic/`；
- `outputs/phase2d_c8_seed303_video_freeze/`；
- `outputs/phase2d_c8_seed303_geometry_freeze/`；
- `outputs/phase2d_c8_seed303_candidate_gate_freeze/`。

以下来源会被拒绝：

- Ground Truth 和 GT evaluation；
- `manual_prompt`、`blind_*` 和 WSL SAM 2 临时工作区；
- `water_test`；
- 宿舍、纸箱或 cardboard/dormitory 素材；
- 允许目录以外的任意绝对路径或路径穿越。

所有演示结果继续保持：

- `ground_truth_used=false`；
- `authoritative=false`；
- `eligible_for_downstream=false`。

## 模块边界

### 比赛候选主链路

动态仿真 Camera 视频 → C6 时序证据自动提示 → SAM 2 锚点 mask → C7 视频传播 → 岸线与 Ground DEM → 水位/水深/面积/体积 → C8 candidate gate → C9 canonical shadow。

### 离线研发工具

Gazebo/动态降雨生成、Ground Truth、独立评价和多 seed 回归只服务研发和比赛证据复现，不进入未来真实设备运行包。

### 历史对照

早期纯规则时序 water mask、人工提示 SAM 2、configured depth 和 surface DEM 直接相减继续保留为消融与历史对照，不作为比赛默认主链路，也不删除源码。

## 运行

比赛时使用独立入口，它会先生成只读展示快照，再启动只包含仿真道路页面的 Dashboard：

```bash
bash scripts/run_phase2d_c10_competition_demo.sh
```

如只需要校验和生成快照，不启动 Dashboard：

```bash
python3 scripts/build_phase2d_c10_competition_demo.py
```

完整 Dashboard 仍保留“比赛演示：仿真道路积水闭环”页面用于研发检查，但比赛专用入口不显示宿舍纸箱等历史对照页面。两种入口都不启动 ROS、Gazebo、Camera、LiDAR、RTSP 或正式 Agent，也不生成正式预警。
