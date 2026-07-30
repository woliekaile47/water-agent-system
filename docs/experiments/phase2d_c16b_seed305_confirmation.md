# Phase 2D-C16B：seed305 独立确认实验

## 1. 实验目的

本阶段使用全新的 seed305，对 Phase 2D-C16 已冻结的 5 cm 浅水时序稳定化方案进行第二次独立确认。

本阶段只回答两个问题：

1. 已冻结方案能否在新 seed 上继续将水位误差控制在 3 cm 内；
2. prediction-side 质量门能否阻止不可信结果进入后续链路。

本阶段不是新的调参阶段。实验前已固定：

- 水深：5 cm；
- rain level：light、moderate、heavy；
- seed：305；
- 帧范围：129–169；
- anchor frame：149；
- 每组帧数：41；
- 自动时序提示算法；
- SAM 2 模型与传播方法；
- 5 帧、80% 支持的 mask 稳定化规则；
- Camera–DEM 几何反演；
- 全部质量门配置和阈值。

未根据 seed305 的 RGB、prediction 或 Ground Truth 结果修改任何参数。

## 2. Prediction 与 Ground Truth 隔离协议

seed305 严格按照“先 prediction、后 evaluation”的顺序执行。

### 2.1 Prediction 阶段

Prediction 只读取：

- 固定 RGB 帧；
- 自动时序提示；
- SAM 2 模型与冻结传播结果；
- dry Ground DEM；
- Camera 参数；
- mask-to-DEM mapping 配置；
- 现有 prediction-side quality gate。

Prediction 不读取：

- Camera water mask Ground Truth；
- water level Ground Truth；
- DEM water mask Ground Truth；
- depth map Ground Truth；
- area 或 volume Ground Truth；
- 任何既有 evaluation 输出。

三个序列均只执行一次有效 prediction。完成后冻结：

- 自动 prompt；
- SAM 2 mask；
- mask 稳定化结果；
- water level、depth、area 和 volume prediction；
- prediction-side quality gate；
- 源码、配置、输入与输出 SHA-256。

动态序列第一次生成尝试因没有加载 ROS 2 Python 环境，在读取基础 RGB 前停止。随后仅加载 `/opt/ros/humble` 环境并使用完全相同的 seed 和配置生成数据；该基础设施重试发生在任何 prompt、SAM 2、geometry prediction 或 GT evaluation 之前，没有改变实验选择和参数。

### 2.2 独立 Evaluation 阶段

只有在全部 prediction 输出冻结并记录哈希后，独立 evaluation 才首次读取 seed305 Ground Truth。

Evaluation 未进行以下操作：

- 修改自动 prompt；
- 重新运行 SAM 2；
- 重新运行 geometry prediction；
- 根据 GT 重新选择 basin、seed 或 water level；
- 根据评价结果调整算法或阈值。

评价完成后，prediction 主摘要 SHA-256 保持不变。因此，seed305 结果属于一次冻结后的独立确认，而不是查看 GT 后反复优化得到的结果。

## 3. 自动 Prompt 状态

seed305 的 light、moderate、heavy 三组自动 prompt 均为：

```text
diagnostic_only
```

这意味着自动时序视觉证据不足以将提示认定为正式可用输入。

为完成离线研究，三组 `diagnostic_only` prompt 仍被允许进入非权威的 SAM 2 propagation 和 geometry diagnostic，但始终保持：

```text
authoritative = false
eligible_for_downstream = false
```

因此，即使某一帧的 geometry quality gate 显示 `pass`，也不代表端到端链路已经可用。完整链路必须继承上游 prompt 的较低状态。本阶段 seed305 的端到端状态均为 `diagnostic_only`。

## 4. seed304 与 seed305 对比

### 4.1 Camera mask、水位精度与稳定性

| Seed | 雨强 | raw SAM 2 Camera IoU 中位数 | 水位误差中位数 | 水位误差最大值 | 水位误差 ≤ 3 cm | 水位时间标准差 | Geometry pass |
|---:|---|---:|---:|---:|---:|---:|---:|
| 304 | light | 0.8761 | 0.9328 cm | 1.1618 cm | 41/41 | 0.1040 cm | 0/41 |
| 304 | moderate | 0.8781 | 0.7033 cm | 1.0784 cm | 41/41 | 0.1665 cm | 19/41 |
| 304 | heavy | 0.5759 | 1.4814 cm | 4.8489 cm | 34/41 | 1.0338 cm | 0/41 |
| 305 | light | 0.8480 | 1.0477 cm | 1.3316 cm | 41/41 | 0.1594 cm | 0/41 |
| 305 | moderate | 0.8784 | 0.8351 cm | 1.1590 cm | 41/41 | 0.1238 cm | 7/41 |
| 305 | heavy | 0.6570 | 1.2385 cm | 2.3452 cm | 41/41 | 0.3706 cm | 0/41 |

seed304 共 116/123 帧的水位误差不超过 3 cm。

seed305 共 123/123 帧的水位误差不超过 3 cm，且三个雨强的 41 帧水位时间标准差均低于 1 cm。

seed305 没有出现“geometry gate 通过但水位误差超过 3 cm”的错误放行：

```text
bad_geometry_pass_count = 0
```

但这不能解释为 seed305 已经实现端到端可用，因为三个自动 prompt 均为 `diagnostic_only`。

### 4.2 Prediction-side 自一致性

| 雨强 | Camera 重投影 IoU 中位数 | 参考 boundary P95 中位数 | Geometry pass / reject |
|---|---:|---:|---:|
| light | 0.8573 | 2.828 px | 0 / 41 |
| moderate | 0.8887 | 2.236 px | 7 / 34 |
| heavy | 0.7868 | 8.062 px | 0 / 41 |

主要拒绝原因仍是 `camera_reprojection_iou_below_threshold`。heavy 另有 1 帧触发 `shoreline_height_mad_above_threshold`。

这些指标由 prediction 自身计算，不读取 Ground Truth。

## 5. Camera 可见区域面积和体积

以下误差仅评价 Camera 可见主盆地，不代表整段道路的全局积水总量。

| 雨强 | Camera 可见面积相对误差中位数 | Camera 可见体积相对误差中位数 |
|---|---:|---:|
| light | 12.03% | 21.91% |
| moderate | 6.96% | 12.56% |
| heavy | 15.19% | 30.52% |

水位误差较小并不意味着面积和体积同样准确。面积与体积对 Camera mask 边界更加敏感，尤其在 shallow water 和 heavy rain 条件下，边界少量扩张或收缩会产生较明显的面积、体积误差。

这些结果只能表述为：

```text
camera_visible_candidate_estimate
```

不得表述为全局道路积水测量结果。

## 6. 结果分析

### 6.1 水位安全性得到复现

seed305 三种雨强共 123 帧全部满足水位误差不超过 3 cm，时间标准差也全部低于 1 cm。

结合 seed304 结果可以确认：

- light 和 moderate 的水位误差在两个 seed 上均稳定低于 3 cm；
- seed304 heavy 中误差超过 3 cm 的帧被质量门全部拒绝；
- seed305 没有出现误差超过 3 cm 的帧；
- 两个 seed 均未出现不可信水位被 geometry gate 错误放行。

因此，本阶段的安全性确认结果为：

```text
safety_confirmation = confirmed
```

这里的“安全性确认”表示系统能够做到“误差满足目标，或在证据不足时拒绝”，不表示每一帧都能产生正式可用结果。

### 6.2 可用性尚未确认

seed304 的 moderate 有 19/41 帧通过 geometry gate，seed305 的 moderate 只有 7/41 帧通过。light 和 heavy 在两个 seed 上均没有 geometry pass。

更重要的是，seed305 三组自动 prompt 全部为 `diagnostic_only`。因此，moderate 的 7 个 geometry pass 也不能升级为端到端正式 pass。

本阶段的可用性结论为：

```text
availability_status = limited
availability_confirmation = not_confirmed
```

当前系统已证明能够在离线仿真中给出精度较好的水位候选，并能拒绝部分不可信结果，但尚未证明自动时序提示能够稳定地产生正式可用输入。

### 6.3 Heavy 场景有所改善，但仍不能放行

seed305 heavy 相比 seed304 heavy：

- Camera IoU 中位数由 0.5759 提高到 0.6570；
- 水位最大误差由 4.8489 cm 降至 2.3452 cm；
- 水位时间标准差由 1.0338 cm 降至 0.3706 cm。

这些指标表明 seed305 heavy 的离线预测优于 seed304 heavy，但 Camera mask IoU 仍明显低于 light 和 moderate，且自动 prompt 仍是 `diagnostic_only`，geometry 也为 0/41 pass。

因此不能把 seed305 heavy 描述为已经解决。

## 7. 阶段结论

Phase 2D-C16B 的准确结论为：

1. seed305 在完全冻结参数下，123/123 帧水位误差均控制在 3 cm 内；
2. seed305 三种雨强的水位时间标准差均低于 1 cm；
3. seed304 中超过 3 cm 的 heavy 错误结果没有被质量门放行；
4. seed305 的错误 geometry pass 数量为 0；
5. 水位精度与安全拒绝机制在第二个新 seed 上得到复现；
6. seed305 三组自动 prompt 均为 `diagnostic_only`；
7. moderate 的 7 个 geometry pass 仍不能视为端到端正式可用；
8. light 和 heavy 仍全部被 geometry gate 拒绝；
9. Camera 可见面积和体积误差仍高于水位误差；
10. 当前系统安全性得到确认，但自动化可用性尚未确认。

最终状态：

```text
prediction_accuracy_status = partially_confirmed
safety_confirmation = confirmed
availability_confirmation = not_confirmed
authoritative = false
eligible_for_downstream = false
```

## 8. 为什么不能放宽阈值

seed305 的评价结果不能用于降低现有质量门阈值，原因包括：

1. seed305 已被用于独立确认，不能再作为阈值选择集；
2. 自动 prompt 全部为 `diagnostic_only`，主要问题位于上游视觉证据，而不是单一 geometry 阈值；
3. light 虽然水位准确，但仍存在面积和边界不可信；
4. heavy 的 Camera mask IoU 仍明显偏低；
5. 放宽 geometry 阈值可能把未来其他 seed 的真实视觉失败放行；
6. 当前仅有 seed304 和 seed305，样本规模不足以重新设计正式质量门。

下一步应改进自动时序提示的证据完整性，而不是根据 GT 结果降低阈值。

## 9. 下一步建议

1. 冻结 seed304 和 seed305，不再使用其 GT 调整 prediction。
2. 只读分解三个自动 prompt 为 `diagnostic_only` 的 prediction-side 原因。
3. 优先检查正点、负点、box 以及上游时序证据的哪一项触发降级。
4. 保持统一参数，禁止为 light、moderate、heavy 分别调参。
5. 如需验证新的提示方案，必须使用全新 seed。
6. 分别报告 prompt、geometry、water-level、Camera-visible area/volume 和 global-scene 状态。
7. 在自动 prompt 可用性得到独立验证前，不接入正式 S5–S8。

## 10. 合规声明

本阶段：

- 未在 seed305 上逐 case 调参；
- 未修改 prediction 算法；
- 未修改质量门阈值；
- 未使用 GT 修改 prompt、SAM 2 mask、water level、seed 或 basin；
- prediction 输出先冻结，随后才进行独立 GT evaluation；
- 未重新运行 seed305 prediction；
- 未把 geometry pass 冒充端到端 pass；
- 未把 `diagnostic_only` 结果标记为 authoritative；
- 未接入 S5–S8、Agent、数据库或正式预警；
- 未启动真实 Camera、LiDAR、RTSP、ROS 节点或 Gazebo；
- 所有结果均为仿真环境下的非权威研究输出。
