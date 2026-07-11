# Phase 2C-1 Dynamic Rain Visual Simulation

## 1. 本阶段目的

Phase 2C-1 生成可重复的连续 RGB 图像序列，用受控的时域视觉差异表达雨滴落在普通路面与积水表面时的不同现象，为 Phase 2C-2 仅依赖视频帧的积水视觉识别提供训练和离线评价数据。

本阶段只实现 synthetic sequence generator，不实现雨滴检测器、Camera water mask 预测、神经网络训练、S5-S8、Agent、数据库或 API。

## 2. 为什么不直接训练检测器

当前还没有经过整理和标注的真实连续降雨视频。如果直接开发检测器，会同时引入数据生成正确性、时域特征设计和识别算法三类不确定性，无法区分错误来自哪里。

Phase 2C-1 先固定数据接口和可控事件语义，并验证：

- dry splash 是短生命周期、非连续同心扩张；
- water ripple 有较长生命周期、扩张环和衰减振幅；
- 连续帧、事件 Ground Truth 和 water mask Ground Truth 能严格分离；
- 相同 seed 的序列逐帧一致；
- light、moderate、heavy 的事件数量严格递增。

Phase 2C-2 才允许实现只读取 `frames/` 的视觉算法。

## 3. 基础 RGB 图像来源

基础图像来自现有 Phase 1A dry rosbag：

```text
data/simulation/sim_dry_baseline_001/rosbag/
  sim_dry_baseline_001__20260710T185621Z/
```

读取 topic：

```text
/sim/camera/image_raw
sensor_msgs/msg/Image
encoding: rgb8
resolution: 640 × 360
```

生成器使用 `rosbag2_py.SequentialReader` 以只读方式打开 SQLite rosbag，不执行 rosbag replay，不启动 Gazebo 或 ROS 实时节点。它跳过前 10 个 Camera 帧，对后续 5 帧做逐像素中位数，得到稳定、确定性的 dry road 基础图。

基础 RGB 不依赖人工截图。所有场景共用同一张无水道路基础图，water case 再由生成器内部依据各自 Camera GT mask 添加轻量水面外观和动态涟漪。

## 4. 目录与模块边界

生成器位于：

```text
src/perception/synthetic_rain_visual_generator.py
```

选择 `src/perception/` 是为了把视觉数据生成和未来视觉检测接口放在同一领域层，同时不修改 Phase 1A 的 Gazebo package、静态场景、LiDAR 或 Camera 驱动。

每个 sequence 的输出严格分成：

```text
frames/         # 未来 detector 唯一允许读取的 RGB 帧
ground_truth/   # water mask、事件类别、活动图和时域状态
metadata/       # manifest、配置快照、质量报告
```

`frames/` 中只包含连续 PNG，不包含 Ground Truth 文件或快捷链接。

## 5. 干地飞溅视觉抽象

`dry_splash` 模型包括：

- 事件中心只能从非水区域采样；
- 生命周期为 2～5 帧；
- 局部放射状细线模拟不规则飞溅；
- 少量短距离 satellite droplets；
- 半径和强度由统一雨强配置采样；
- 峰值后指数快速衰减；
- 不产生持续扩张的稳定同心环；
- 图像边缘使用安全裁剪，不发生数组越界。

每个事件记录 `event_id`、类型、中心、开始/结束/峰值帧、半径、强度、spoke 数、satellite 数和独立 random seed。

## 6. 水面撞击与涟漪视觉抽象

`water_ripple` 模型包括：

- 事件中心只能从 water mask 内部采样；
- 初始小尺度冲击亮斑或暗斑；
- 一个或多个旋转椭圆环；
- 半径按 `initial_radius + expansion_rate × age` 扩张；
- 振幅按 `intensity × damping_factor^age` 衰减；
- 生命周期明显长于 dry splash；
- 多事件允许时域和空间重叠；
- 叠加局部水面反射扰动；
- 环不会硬裁剪成 Ground Truth water mask 形状。

为了让主要能量仍与水域相关，water mask 只作为软能量权重：mask 外仍保留配置比例的环能量，避免产生人为的水域边界切口。

每个事件记录初始半径、扩张速度、damping、ring count、ellipse ratio、旋转角度、强度、生命周期和 random seed。

## 7. Camera 与环境扰动

当前统一配置支持：

- 缓慢全局亮度变化；
- 轻微曝光波动；
- 高斯图像噪声；
- 确定性雨丝和简化运动拖影；
- water mask 内的局部反射闪烁；
- 下采样/最近邻回放混合形成的简化压缩块效应；
- 可选 Camera jitter，默认关闭以避免 RGB 与静态标注错位。

所有扰动均有开关和强度参数，并由 sequence seed 与 frame index 确定。扰动不会修改 Ground Truth water mask。

## 8. Ground Truth 使用边界与防泄漏

Camera Ground Truth water mask 只在 synthetic generator 内部用于：

- 区分 dry splash 与 water ripple 的合法中心；
- 添加水面基础外观和软反射权重；
- 生成事件类别、生命周期和活动区域标注；
- 数据生成质量检查。

Ground Truth 输出包括：

- `water_mask.png`；
- `observable_water_mask.png`；
- `event_map_sequence.npy`，bit 1 表示 dry splash，bit 2 表示 water ripple；
- `temporal_activity_map.npy`；
- `event_annotations.json`；
- `frame_event_counts.json`；
- `event_states.json`，记录每帧活动事件的中心、半径和振幅。

Manifest 明确包含：

```text
ground_truth_used_for_generation_only: true
detector_input_should_only_use_frames: true
synthetic_physics_validity: visual_abstraction_not_fluid_simulation
```

未来 Phase 2C-2 推理代码不得读取 `ground_truth/`、`metadata/`、事件类别、生成参数或 case 真实水深。

## 9. 数据集组合与数据量

统一生成参数：

```text
duration: 10 s
fps: 20
frames per sequence: 200
resolution: 640 × 360
```

组合为 5 个场景 × 3 个雨强 × 1 个固定 seed，共 15 个 sequence、3000 帧，磁盘占用约 1.7 GiB。

| Rain level | Seed | Sequence | Frames | Dry events | Water events | Total events |
|---|---:|---:|---:|---:|---:|---:|
| light | 41 | 5 | 1000 | 59 | 16 | 75 |
| moderate | 42 | 5 | 1000 | 128 | 72 | 200 |
| heavy | 43 | 5 | 1000 | 224 | 176 | 400 |

Dry baseline 的 water event 始终为 0。四个 water case 在相同雨强下使用相同总事件数和类别比例，避免逐 case 调参。

## 10. 随机种子与可重复性

事件表、中心、生命周期、事件参数、雨丝和 Gaussian noise 都从固定 seed 派生。每个事件另有独立 seed，使事件图形可重复且不依赖渲染顺序。

`sim_water_10cm_001 / moderate / seed_42` 的完整 200 帧重复生成两次，SHA-256 均为：

```text
05bf09968fc0ecce842e63b9a6d223903c9bee6c74d0c01eb1e3ec1616a9efc9
```

自动测试还确认相同 seed 逐帧 PNG 字节一致，不同 seed 至少部分帧不同。

## 11. 10 cm Moderate 样例

```text
case: sim_water_10cm_001
rain: moderate
seed: 42
frames: 200
dry events: 22
water events: 18
mean dry lifetime: 3.45 frames
mean water lifetime: 21.39 frames
quality: pass
```

关键帧检查表明：dry splash 短促且不形成持续同心环；water ripple 从冲击点扩张为椭圆环并逐步衰减；扩张环未被硬裁剪成 water mask 形状。道路、路沿和水域仍然可辨认，但画面保留明显的可控合成特征。

## 12. 质量检查结果

15 个 sequence 的 `generation_quality_report.json` 全部为 pass：

- sequence 数：15；
- 帧数：3000；
- 帧编号连续；
- 分辨率全部为 640×360；
- 所有帧无 NaN/Inf；
- 全数据像素范围为 21～255；
- 所有 dry 中心位于非水区域；
- 所有 water 中心位于 water mask；
- dry scene 无 water ripple；
- dry 平均生命周期短于 water；
- ripple 半径扩张且振幅衰减；
- 同 seed schedule 一致、不同 seed 不同；
- light < moderate < heavy；
- Ground Truth 文件完整；
- `frames/` 无 Ground Truth symlink。

数据集汇总保存在：

```text
data/simulation_dynamic/dataset_summary.csv
data/simulation_dynamic/dataset_summary.md
```

自动生成数据由 `.gitignore` 排除，只保留 `data/simulation_dynamic/.gitkeep`。

## 13. 物理真实性限制

本阶段是可控的视觉抽象仿真，不是经过真实雨滴实验验证的流体动力学模型。

当前模型没有计算真实液滴尺寸谱、落速、表面张力、粘性、风场、真实路面微结构、波传播方程或多尺度流体耦合。椭圆环、放射状飞溅、衰减和反射只是为了构造可测试的时域差异。

仿真数据不能替代真实降雨场景验证，也不能直接证明模型能够部署。后续必须使用真实降雨视频进行域适配、误差分析和独立验收。

## 14. Phase 2C-2 计划

Phase 2C-2 应保持 detector 与生成器完全分离：

1. detector 只读取 `frames/`；
2. 先实现不读取 Ground Truth 的时域特征基线；
3. 比较短促非环状 dry splash 与长生命周期扩张 ripple；
4. 推理完成后由独立 evaluator 读取 Ground Truth；
5. 对反射、夜间、暴雨雨幕、Camera 抖动和压缩失真做鲁棒性验证；
6. 引入真实降雨视频进行域适配；
7. 在真实验证和 quality gate 完成前，不接入 S5-S8 或正式 Agent 链路。
