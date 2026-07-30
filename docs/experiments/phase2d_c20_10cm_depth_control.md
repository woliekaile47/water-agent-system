# Phase 2D-C20：10 cm 深度受控对照实验

## 实验目的

本实验用于判断 Phase 2D-C19 的 5 cm 自动视觉失败，是否主要来自浅水视觉证据不足。

C20 与 C19 保持以下条件完全一致：

- 中雨（moderate）；
- 随机种子 308；
- 60 秒、20 FPS，共 1201 帧；
- 固定 41 帧密集窗口；
- 相同的自动窗口选择、时序检测、提示生成、SAM 2 视频传播、mask 稳定化、几何反演和质量门；
- 不使用人工提示；
- 不允许失败后人工兜底；
- prediction 冻结前不读取 Ground Truth。

唯一实验变量是场景由 `sim_water_5cm_001` 改为 `sim_water_10cm_001`。

## 预测侧流程

1. 生成 `sim_water_10cm_001/moderate/seed_308` 的 60 秒动态序列。
2. 在 13 个固定候选窗口中，使用不读取 GT 的固定排序规则选择一个 41 帧窗口。
3. 仅当自动提示质量状态为 `pass` 时，允许运行一次 SAM 2。
4. 对冻结的 41 帧 SAM 2 mask 运行现有几何反演和 prediction-side quality gate。
5. 冻结全部输出和 SHA-256 后，才由独立 evaluation 模块读取 GT。

自动窗口选择结果：

- selection status：`pass`
- selected candidate index：6
- SAM 2 allowed：`true`
- positive points：3
- negative points：8
- manual selection：`false`
- Ground Truth used：`false`

SAM 2 视频传播结果：

- frame count：41
- adjacent mask IoU median：0.968397
- mask area coefficient of variation：0.014282
- elapsed time：7.508 s

预测侧几何结果：

- geometry available：41/41
- quality status：41/41 `pass`
- estimated water level median：-0.351011 m
- maximum depth median：9.2883 cm
- area median：3.02 m²
- volume median：0.138198 m³
- Camera reprojection IoU median：0.927016
- boundary reprojection P95 median：2.236 px
- adjacent water-level change P95：0.1341 cm

以上结果均为非权威候选，`eligible_for_downstream=false`。

## 独立 Ground Truth 评价

评价模块在确认预测输出哈希后独立读取 GT；没有重新运行 SAM 2、没有重新计算 prediction、没有修改提示或质量门。

Camera mask：

- IoU：中位数 0.933735，最小 0.906547，最大 0.949986
- precision：中位数 0.984735
- recall：中位数 0.950747
- F1：中位数 0.965732
- outer boundary P95：中位数 3.0 px
- 严格离线研究条件满足：37/41 帧

水位与几何：

- 水位绝对误差：中位数 0.7117 cm，最大 0.9417 cm
- 3 cm 目标内：41/41 帧
- 可见面积相对误差：中位数 7.08%
- 可见体积相对误差：中位数 13.94%
- 可见平均水深绝对误差：中位数 0.3648 cm
- 最大水深绝对误差：中位数 0.7117 cm

10 cm 场景只有一个 Camera 可观测积水盆地，因此本实验中的可见结果与全局场景结果一致。

## 与 C19 5 cm 的受控对照

| 项目 | C19：5 cm | C20：10 cm |
|---|---:|---:|
| rain / seed / duration | moderate / 308 / 60 s | moderate / 308 / 60 s |
| 固定候选窗口数量 | 13 | 13 |
| 自动选择状态 | reject | pass |
| 安全正点数量 | 0 | 3 |
| 是否允许运行 SAM 2 | 否 | 是 |
| 预测侧几何有效帧 | 0 | 41/41 |
| 水位误差小于 3 cm | unavailable | 41/41 |
| Camera mask IoU 中位数 | unavailable | 0.933735 |

C19 的独立诊断显示：5 cm 选中区域虽然 precision=1.0，但 recall 仅为 0.0477，只覆盖真实水域中的极小核心，无法安全地产生足够正点。

## 结论

本受控实验支持以下判断：

1. 当前自动时序提示与 SAM 2 链路在 10 cm 场景能够稳定工作；
2. 5 cm 失败并非由 Camera 几何、ray–DEM 求交或统一质量门普遍失效造成；
3. 当前主要瓶颈是 5 cm 浅水在动态雨滴、反射和湿路背景下提供的可分割视觉证据不足；
4. 不应通过放宽安全提示门或使用 GT 调参强行让 5 cm 进入 SAM 2；
5. C20 仍是仿真、非权威离线研究结果，不能直接接入正式 S5–S8。

## 冻结结果路径

- 自动窗口：`outputs/temporal_dense_burst_c20_10cm_60s_seed308/`
- SAM 2 传播：`outputs/sam2_video_pilot_c20_10cm_60s_seed308/`
- 几何预测：`outputs/sam2_video_geometry_stability_c20_10cm_60s_seed308/`
- Camera GT 评价：`outputs/sam2_video_propagation_gt_evaluation_c20_10cm_60s_seed308/`
- 几何 GT 评价：`outputs/sam2_video_geometry_gt_evaluation_c20_10cm_60s_seed308/`

本阶段未启动真实设备、ROS 节点或 Gazebo，未执行 Git commit 或 push。
