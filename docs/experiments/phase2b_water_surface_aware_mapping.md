# Phase 2B Water-surface-aware Camera Mask Mapping 实验

## 1. 实验目的与初始假设

Phase 2A 使用无水 Ground DEM 栅格点反向投影 Camera mask。5、10、20 cm 场景结果良好，但 40 cm 的全局评价为：

```text
IoU = 0.8354
Recall = 0.8358
面积误差 = 3.62 m²
```

Phase 2B 开始时的假设是：Camera 实际观察水平水面，而 Phase 2A 投影的是更低的水下地面；高水位时水面与地面之间视差增大，导致 Camera mask 映射到 DEM 时漏掉浅水边缘。

因此 Phase 2B 实现了 image-space shoreline、Camera ray、ray–DEM 求交、shoreline 水位估计、低地连通重建和水面重投影自一致性，目标是在不读取 Ground Truth 答案的前提下修复高水位误差。

## 2. 实际排查结论

实际数据表明，初始视差假设不是 40 cm 全局 recall 偏低的主要原因。

40 cm Ground Truth DEM mask 包含两个彼此独立的连通盆地：

| Ground Truth 水域 | Cell 数 | Phase 2B intersection | Recall | Camera 水面投影 |
|---|---:|---:|---:|---:|
| 可见主水域 | 1848 | 1848 | 1.0 | 可见 |
| 不可见第二水域 | 363 | 0 | 0.0 | 0 pixels |

可见主水域已经被完整覆盖。第二水域的水平水面投影在当前 Camera 中为 0 像素，且没有 Camera seed 或 Phase 2A ground-point seed，因此它在允许的预测输入中完全不可观测。

40 cm 未达到全局目标不是一个能够通过继续调参安全解决的问题，而是单 Camera 观测范围不足造成的不可观测性问题。

## 3. 为什么不自动填充第二盆地

在估计水位下，Ground DEM 确实会产生第二个低于水位的候选盆地。但“地形低于估计水位”只能说明该区域可能容纳积水，不能证明该独立盆地已经进水并与可见主水域具有相同水位。

第二盆地同时满足：

- Camera 投影像素为 0；
- 无 Camera mask 支持；
- 无有效 seed；
- 与主水域在 DEM 上不连通。

强行把它加入 predicted DEM mask 会构成无观测依据的猜测，也可能在真实道路上把无关低地误报为积水。因此 Phase 2B 不自动填充该盆地，而是把全局结果标记为 `partial`，将面积和体积解释为 Camera 可见范围下界，并由全局 quality gate reject。

## 4. Ray–DEM 求交

Phase 2B 的 shoreline 几何处理为：

1. 从 Camera 二值 mask 提取水像素与非水像素的亚像素界面；
2. 使用像素边中点，而不是水域内部边界像素中心，减少向水域内部的几何偏差；
3. 使用 CameraInfo 将像素转换为 `camera_optical_frame` 射线；
4. 光学坐标遵循 X 向右、Y 向下、Z 向前；
5. 将射线转换到 `map`；
6. 沿射线步进，使用 Ground DEM 双线性插值计算 `z_ray-z_dem`；
7. 检测残差符号变化后进行二分细化；
8. 输出像素、map 交点、DEM 高程、射线距离和最终残差；
9. 记录无交点、越界和其他失败原因。

四个场景的 shoreline ray 求交成功率均为 1.0。

## 5. Shoreline 水位估计

统一配置使用 `robust_median`：

- 按 Camera mask 连通域分别处理；
- 过滤 NaN/Inf；
- 使用 MAD 剔除离群点；
- 支持 `median`、`trimmed_median` 和 `robust_median`；
- 输出样本数、MAD、IQR、标准差及估计水位；
- 不读取真实水位，不使用真实水位选择参数。

当前水位由确定性 ray–DEM shoreline 直接求解，不进行 Ground Truth 引导的优化。输出记录一次直接求解、零增量和 `water_level_converged=true`。

## 6. DEM 水域重建

水位估计后生成：

```text
candidate_lowland = finite_ground_dem AND ground_dem < estimated_water_level
```

随后：

1. Phase 2A ground-point mask 仅作为 flood-fill seed；
2. 每个候选盆地按估计水位重投影到 Camera；
3. 有 seed 或得到 Camera mask 明确支持的候选盆地可以保留；
4. flood fill 只在低于水位的连通区域中传播，高地或路沿形成阻断；
5. 无 seed、无 Camera 投影的候选盆地不会自动加入 predicted mask；
6. 这类盆地被记录为 unobservable/ambiguous candidate，并触发全局 reject。

## 7. Phase 2A / Phase 2B 对比

| Case | 2A IoU | 2B IoU | 2A Recall | 2B Recall | 2A 面积相对误差 | 2B 面积相对误差 | 2B Reprojection IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 cm | 1.0000 | 0.9937 | 1.0000 | 1.0000 | 0.000% | 0.633% | 0.9888 |
| 10 cm | 1.0000 | 0.9939 | 1.0000 | 1.0000 | 0.000% | 0.615% | 0.9904 |
| 20 cm | 1.0000 | 0.9888 | 1.0000 | 1.0000 | 0.000% | 1.130% | 0.9809 |
| 40 cm | 0.8354 | 0.8302 | 0.8358 | 0.8358 | 16.373% | 15.739% | 0.9890 |

5、10、20 cm recall 均保持 1.0。Phase 2B 的少量 IoU/面积变化来自亚像素 shoreline 水位和栅格边缘差异。40 cm 可见主水域 recall 为 1.0，但全局 recall 受到不可见第二水域限制。

## 8. Prediction-side 自一致性与 Quality Gate

Quality gate 只使用 prediction-side 指标：

- shoreline 求交成功率；
- Camera mask 图像边缘触碰率；
- 水面重投影 IoU；
- 边界重投影距离；
- 水面投影覆盖率；
- 候选盆地、可观测盆地和不可观测盆地数量；
- seed 有效性；
- 水位收敛状态；
- 水位和 depth 的有限性及物理上限；
- 输出文件完整性。

Quality gate 不接收真实 water level、DEM GT mask、depth GT、真实面积、真实体积、Ground Truth IoU 或 recall。

| Case | Gate | Reasons |
|---|---|---|
| 5 cm | pass | `[]` |
| 10 cm | pass | `[]` |
| 20 cm | pass | `[]` |
| 40 cm | reject | `ambiguous_candidate_basin`; `candidate_basin_outside_camera_coverage` |

40 cm 虽然可见区域自一致性良好（Camera reprojection IoU 约 0.9890），但仍存在一个完全不可观测的候选盆地，因此全局结果必须 reject。

## 9. 结果语义

Phase 2B 结果明确区分 Camera 可观测区域估计和整段道路全局估计：

```yaml
observation_scope: camera_observable_region
global_estimate_status: complete | partial | unavailable
observable_region_result_valid: true | false
unobservable_candidate_basin_count: <int>
ambiguous_candidate_basin_count: <int>
camera_observable_candidate_basin_count: <int>
result_semantics: observable_region_estimate | global_estimate
area_volume_semantics: observable_lower_bound | complete_estimate
eligible_for_downstream: false
```

5、10、20 cm 没有不可观测候选盆地，标记为 `complete`、`global_estimate` 和 `complete_estimate`。

40 cm 标记为：

```yaml
observation_scope: camera_observable_region
global_estimate_status: partial
observable_region_result_valid: true
unobservable_candidate_basin_count: 1
ambiguous_candidate_basin_count: 1
camera_observable_candidate_basin_count: 1
result_semantics: observable_region_estimate
area_volume_semantics: observable_lower_bound
eligible_for_downstream: false
```

因此，40 cm 输出的面积和体积只能解释为 Camera 可见范围内的下界，不能冒充整段道路的完整积水总量。

## 10. 当前系统的可观测性边界

单 Camera 系统只能对其视锥内、能够形成 Camera mask 或可验证重投影的水域负责。仅凭一个可见水域的水位和道路 DEM，无法判断视野外、与可见水域不连通的独立低地是否已经积水。

这一边界不能通过调低 quality gate 阈值、改变分位数或自动填充所有低地安全解决。继续针对 40 cm 全局 IoU 调参会把不可观测性问题伪装成算法精度问题。

## 11. 后续解决方向

要获得整段道路的完整全局估计，应扩大实际观测覆盖，而不是猜测不可见区域：

1. 调整 Camera 安装位置、俯角或镜头视场，覆盖第二盆地；
2. 使用多 Camera，并完成跨 Camera 外参标定与 mask 融合；
3. 对关键盲区增加水尺、液位计、毫米波雷达或其他水位传感器；
4. 使用具有物理连通信息的排水/路面拓扑模型，但必须把模型推断与直接观测明确区分；
5. 在 Dashboard 中同时展示 Camera 可观测范围、全局完整性状态和面积体积语义；
6. 在正式接入 S5-S8 前继续保持 `eligible_for_downstream=false`。
