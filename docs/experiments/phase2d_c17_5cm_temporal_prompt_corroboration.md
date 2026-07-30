# Phase 2D-C17：5 cm 自动提示多时段旁证与 seed306 验证

## 1. 目的

Phase 2D-C16B 在 seed305 上确认了 5 cm 水位精度和安全拒绝能力，但 light、moderate、heavy 三组自动提示全部为 `diagnostic_only`。

本阶段只解决一个问题：在不读取 Ground Truth、不降低旧质量门阈值的前提下，判断多个时间段重复出现的稳定视觉证据能否支持生成可用的 SAM 2 自动提示。

所有结果仍为非权威研究候选：

```text
authoritative = false
eligible_for_downstream = false
```

## 2. 原因诊断

seed305 的自动提示本身均满足：

- 单一主连通域；
- 5 个安全正点；
- 8 个环绕负点；
- 正点概率不低于 0.50；
- 正点在三个时间窗中的支持率均为 1.0；
- box 不触碰图像边界；
- 无候选组件歧义；
- 未读取 Ground Truth。

`diagnostic_only` 的唯一原因来自上游时序质量门：

```text
insufficient_high_confidence_water_tracks
```

旧质量门要求至少存在一条通用 `confidence >= 0.50` 的水轨迹。5 cm 条件下存在大量被分类为 `water_ripple` 的中等置信轨迹，但没有单条轨迹达到该门槛。提示模块此前无条件继承上游 `partial`，因此即使正点获得三个时间窗共同支持，也不能成为 `pass` 提示。

## 3. 修复方法

本阶段没有修改：

- `configs/temporal_water_quality_gate.yaml`；
- 任何质量门阈值；
- 时序水轨迹分类公式；
- SAM 2 模型或权重；
- Camera–DEM 几何算法；
- S5–S8。

新增的是一个显式启用的“提示级多时段旁证”规则。只有同时满足以下条件时，上游 `partial` 才可在提示层被旁证：

1. 上游不是 `reject`；
2. 上游唯一 partial 原因属于固定白名单，本阶段仅允许 `insufficient_high_confidence_water_tracks`；
3. `observable_region_result_valid=true`；
4. 三时间窗支持图真实存在且数值有限；
5. 正点数量达到原有要求；
6. 每个正点的时间支持率达到固定的 2/3；
7. 提示主域无歧义；
8. box、正点、负点及方向覆盖等原有安全检查全部通过；
9. 不存在任何 hard reject 原因。

该规则只改变 SAM 2 提示候选状态，不改写上游质量门。输出同时保留：

```text
upstream_temporal_quality_gate_status = partial
partial_gate_corroboration_applied = true
```

未启用新配置时，旧行为保持不变。

## 4. Held-out 协议

新验证使用预先冻结的确定性样本：

- 水深：5 cm；
- rain level：light、moderate、heavy；
- seed：306；
- 每组 200 帧；
- anchor frame：149；
- 几何评价窗口：129–169，共 41 帧；
- 选择规则：305 之后的下一个未使用整数 seed；
- 未浏览 RGB 后挑选 seed 或帧；
- 每组自动提示和 SAM 2 propagation 各运行一次；
- prediction 完成并记录 SHA-256 后，独立 evaluation 才读取 GT。

第一次几何命令因协议样本缺少脚本要求的 `role` 字段而停止，没有形成完整结果。该不完整目录被保留。随后仅补充三个 `role: validation` 字段，在新的 `retry1` 目录完成一次有效 prediction；没有改变提示、mask、算法或阈值。

## 5. 自动提示与 SAM 2 传播

| 雨强 | 上游时序门 | 提示旁证 | 自动提示 | 正点/负点 | SAM 2 相邻帧 IoU 中位数 | mask 面积 CV |
|---|---|---|---|---:|---:|---:|
| light | partial | applied | pass | 5 / 8 | 0.9737 | 0.0177 |
| moderate | partial | applied | pass | 5 / 8 | 0.9408 | 0.1271 |
| heavy | pass | not needed | pass | 5 / 8 | 0.9277 | 0.1166 |

seed306 三组提示均为 `pass`。light 和 moderate 通过多时段旁证，heavy 的上游时序门本身已经为 `pass`。

这证明自动提示不再因为旧的单轨迹条件被全部降级，但不代表后续几何结果必然通过。

## 6. 独立 Ground Truth 评价

### 6.1 Camera mask 与水位

| 雨强 | Camera IoU 中位数 | GT 外岸线 P95 中位数 | 水位误差中位数 | 水位误差最大值 | 水位时间标准差 | 误差 ≤ 3 cm |
|---|---:|---:|---:|---:|---:|---:|
| light | 0.8861 | 2.236 px | 0.8855 cm | 1.1467 cm | 0.1331 cm | 41/41 |
| moderate | 0.8153 | 4.499 px | 1.1896 cm | 2.4548 cm | 0.4042 cm | 41/41 |
| heavy | 0.7599 | 6.325 px | 1.3897 cm | 2.5846 cm | 0.3446 cm | 41/41 |

合计：

```text
water_level_within_3cm = 123 / 123
```

三个雨强的水位时间标准差均低于 1 cm，满足本阶段“误差不超过 3 cm 且稳定输出”的研究目标。

### 6.2 Camera 可见面积和体积

| 雨强 | Camera 可见面积相对误差中位数 | Camera 可见体积相对误差中位数 |
|---|---:|---:|
| light | 8.86% | 14.73% |
| moderate | 13.29% | 28.29% |
| heavy | 18.99% | 37.60% |

面积和体积仍明显比水位敏感，尤其在中雨和大雨条件下。它们只能解释为 Camera 可见候选区域估计，不能宣传为全局道路精确测量。

## 7. Prediction-side 质量门

| 雨强 | Camera 重投影 IoU 中位数 | 重投影 boundary P95 中位数 | pass / reject |
|---|---:|---:|---:|
| light | 0.8928 | 2.236 px | 17 / 24 |
| moderate | 0.8720 | 3.162 px | 0 / 41 |
| heavy | 0.8599 | 4.000 px | 0 / 41 |

主要拒绝原因均为：

```text
camera_reprojection_iou_below_threshold
```

17 个通过帧全部满足水位误差不超过 3 cm：

```text
geometry_pass_and_error_above_3cm_count = 0
```

因此，本次提示可用性修复没有造成不可信水位被错误放行。moderate 和 heavy 即使提示为 `pass`，仍被下游几何门保守拒绝。

## 8. 结论

本阶段确认：

1. `diagnostic_only` 的真实原因不是 box、正点或负点失败，而是提示模块无条件继承旧时序门的单轨迹条件；
2. 固定、可解释的多时段旁证可以安全地区分“单轨迹不足但空间证据反复出现”的情况；
3. seed306 三组自动提示均为 `pass`，相比 seed305 的 0/3 有明确改善；
4. seed306 共 123/123 帧水位误差不超过 3 cm；
5. 三组水位时间标准差均低于 1 cm；
6. 没有出现质量门通过但水位误差超过 3 cm；
7. moderate 和 heavy 的 Camera mask、面积及体积仍有明显误差，且几何门仍全部拒绝；
8. 当前只确认了新的自动提示机制在一个全新 seed 上有效，尚不足以宣称真实部署或正式预警可用。

阶段状态：

```text
automatic_prompt_availability = confirmed_on_seed306
water_level_accuracy_and_stability = confirmed_on_seed306
geometry_availability = limited
prediction_accuracy_status = partially_confirmed
authoritative = false
eligible_for_downstream = false
```

## 9. 下一步

不应继续使用 seed304、305 或 306 调整参数。下一步建议在不改参数的情况下：

1. 对 10、20、40 cm 做回归，确认新提示配置不会降低已有能力；
2. 再使用一个全新 seed 做 5 cm 独立确认；
3. 证据足够后再讨论 Camera 重投影门与面积/体积精度；
4. 在真实视频可获得前，不接入正式 S5–S8。

## 10. 合规声明

本阶段未：

- 使用 GT 生成或修改提示；
- 根据 GT 修改 box、正点、负点、SAM 2 mask、水位或 basin；
- 修改旧时序质量门或任何阈值；
- 修改 S5–S8；
- 启动真实 Camera、LiDAR、RTSP、ROS 节点或 Gazebo；
- 把研究候选标为 authoritative；
- 执行 git add、commit 或 push。
