# Phase 2D：WSL 统一运行环境迁移与等价性验证

## 1. 迁移目的

原比赛演示闭环分布在 VMware、Windows 中转目录和 WSL 三个环境中：

1. VMware 使用 ROS 2 处理仿真 LiDAR、构建 Ground DEM，并生成自动 SAM2 提示；
2. Windows 通过 SSH/SCP 中转冻结的图像窗口、提示和 SAM2 输出；
3. WSL 使用 RTX 4060 GPU 执行 SAM2 视频传播；
4. Windows 再将结果送回 VMware；
5. VMware 完成岸线几何、水深反演、质量门、S5–S8 与 Agent。

本阶段将同一链路迁移到 WSL 内部顺序执行，移除运行时对 VMware 和 Windows
中转的依赖。预测算法、SAM2 模型、配置、质量门、安全状态和 S5–S8 数据契约
均保持不变。

## 2. 统一后的运行链路

WSL 本地闭环如下：

```text
仿真 dry LiDAR bag
  → Ground DEM
仿真连续 RGB
  → 时序模块自动生成 SAM2 提示
  → SAM2.1 Hiera Tiny GPU 视频传播
  → Camera water mask
  → 岸线射线与 Ground DEM 求交
  → 水位、深度、面积和体积
  → prediction-side quality gate
  → 允许时进入 S5–S8 与 Agent
```

新增单场景运行器：

```bash
scripts/run_simulation_agent_e2e_wsl.sh
```

新增四水深验收运行器：

```bash
scripts/run_simulation_multiscenario_acceptance_wsl.sh
```

这两个运行器不包含 SSH、SCP 或 VMware 地址，不读取 Ground Truth，不启动真实
Camera/LiDAR，不允许真实预警、外部通知或真实设备动作。

## 3. WSL 环境

- Ubuntu 22.04.5 LTS / WSL2；
- ROS 2 Humble；
- Gazebo Fortress / Gazebo Sim 6.18.0；
- `ros_gz_sim`、`ros_gz_bridge`、`rosbag2`；
- NVIDIA GeForce RTX 4060 Laptop GPU，8 GB 显存；
- SAM 2.1 Hiera Tiny；
- Python 3.10；
- PyTorch 2.7.1 + CUDA 12.8；
- Streamlit 1.58.0；
- NumPy 2.2.6；
- OpenCV 5.0.0。

SAM2 仍使用独立 GPU 虚拟环境，ROS 2 与工程流水线使用系统 Python，避免修改
SAM2 官方仓库或其模型权重。

## 4. 数据迁移

为避免复制整个历史输出目录，只迁移运行与回归所需数据：

- `sim_dry_baseline_001` 无雨 LiDAR 基准；
- 5/10/20/40 cm 的 `moderate/seed_303` 连续 RGB；
- 回归测试清单指定的 12 张 `seed_302/frame_000099` RGB；
- VMware 四场景基线 summary 的只读副本。

共核对 812 个源数据文件，逐文件 SHA-256 比较无差异。没有为预测迁移任何
Ground Truth。

## 5. 回归验证

迁移前未修改工程源码时，WSL 基础回归结果：

- 工程测试：322 passed；
- simulation 测试：14 passed；
- compileall：通过；
- Git 工作区：干净。

新增 WSL 本地运行器后，相关安全、CLI、场景矩阵与 Bash 语法测试也通过。

## 6. 10 cm 最小闭环

运行 ID：

```text
wsl_migration_10cm_20260726_02
```

结果：

- `camera_visible_status = pass`；
- `global_scene_status = complete`；
- Agent 仿真流程完成；
- `estimated_water_level_m = -0.3354649357170738`；
- `mean_depth_cm = 4.983571138355478`；
- `max_depth_cm = 10.492094606161118`；
- `water_area_m2 = 3.2500000000000004`；
- `water_volume_m3 = 0.17154670078307394`；
- `camera_reprojection_iou = 0.9114737883283878`；
- Ground Truth 未用于预测；
- 未启动真实设备；
- 未产生真实预警或外部 API 调用。

以上值与 VMware 基线一致。

## 7. 四水深迁移验收

WSL 矩阵 ID：

```text
wsl_migration_matrix_20260726_01
```

| 场景 | Camera 可见域 | 全局场景 | Agent | 与 VMware 基线 |
|---|---|---|---|---|
| 5 cm | reject | unavailable | 质量门阻断 | 一致 |
| 10 cm | pass | complete | success | 一致 |
| 20 cm | pass | complete | success | 一致 |
| 40 cm | pass | partial | 质量门阻断 | 一致 |

对每个场景比较以下状态字段：

- 流程状态；
- standard pipeline 是否完成；
- Agent 状态和是否应运行；
- Camera 可见域状态；
- 全局场景状态；
- Ground Truth、authoritative 和正式预警资格；
- 真实设备、真实 API 和安全检查状态。

所有状态字段均一致。

对每个场景比较以下数值：

- estimated water level；
- mean depth；
- max depth；
- water area；
- water volume；
- Camera reprojection IoU；
- outer boundary reprojection P95。

在 `1e-12` 比较容差下，四个场景所有数值差值均为 `0.0`。

## 8. 结论

本阶段验证表明：

1. 工程可以在 WSL 中独立完成仿真 LiDAR、自动视觉提示、SAM2、DEM 几何、
   质量门、S5–S8 和 Agent 的完整闭环；
2. WSL 统一运行结果与迁移前 VMware+Windows+WSL 结果完全一致；
3. 5 cm 的拒绝和 40 cm 的全局 partial 是原有可观测性与质量门语义，不是迁移
   故障；
4. VMware 可暂时保留为回滚副本，但不再是比赛演示运行的必要组成；
5. 当前结果仍是仿真、非 authoritative，不代表真实道路实测完成。

## 9. 后续

迁移完成后的下一阶段应优先：

1. 在 WSL 上执行一次人工可复现的离线比赛演示验收；
2. 固化启动说明、依赖版本和结果路径；
3. 再根据明确授权提交并推送迁移代码；
4. 设备回校后进行真实 Camera/LiDAR 标定与分阶段现场验证。
