# Phase 2D-B-1 几何重投影系统误差诊断

## 1. 诊断目的与边界

Phase 2D-A 的 23 条 water sequence 全部触发 `geometry:boundary_reprojection_error_above_threshold`，其中 17 条同时触发 Camera reprojection IoU 低于阈值，另有 8 条 Camera mask IoU ≥ 0.8 但 geometry 仍 reject。本阶段只回答错误更接近固定投影偏移、标定/坐标变换、DEM 重建、视觉岸线不稳定、unknown-aware 域失配还是 3 px 阈值过严。

本阶段严格限定为只读诊断：

- 没有启动 ROS、Gazebo、Camera、LiDAR 或 RTSP；
- 没有重新运行 33 条 prediction，也没有重新运行 temporal detector；
- 没有修改 Phase 2D-A prediction 核心；
- 没有修改任何 gate 阈值；
- 没有修改 GT、正式评价域或既有实验结果；
- 没有把 rejected candidate 改成 authoritative measurement；
- 没有接入 S5–S8、Agent、数据库、FastAPI 或 Dashboard；
- 所有结果保持 `eligible_for_downstream=false`。

## 2. 实际读取的数据

诊断以 `outputs/synthetic_visual_to_depth_integration/dataset_evaluation/dataset_evaluations.json` 中已有的 23 条 water evaluation 为索引，对每条 sequence 只读取既有 artifacts：

- `prediction_manifest.json`：确认 `data_role=prediction`，且 prediction 未读取 GT；
- `predicted_camera_water_mask.png`：时域预测的 observed Camera water mask；
- `predicted_camera_unknown_mask.png`：unknown-aware 评价域；
- `reprojected_camera_mask.png`：DEM candidate water surface 重投影；
- `self_consistency.json`：既有 Camera reprojection IoU、boundary mean/P95；
- `geometry_quality_gate.json`、`visual_quality_gate.json`：状态、原因及时域稳定性；
- `ray_intersection_diagnostics.json`：岸线采样和求交成功率；
- `shoreline_intersections.json`：已有 ray–DEM 交点及数值残差；
- `evaluation_metrics.json` / dataset evaluation：Camera IoU、水位误差和 depth MAE。

诊断脚本没有直接读取或修改原始 GT。水位误差、depth MAE 和 Camera IoU 是 Phase 2D-A 独立 evaluation 已经生成的只读字段。

## 3. 方法

### 3.1 P50/P95 重算

复用冻结的 unknown-aware 定义：

1. observed Camera water mask 与 known non-water 的界面形成 trusted observed shoreline；
2. reprojected mask 先限制到 known region，再形成 trusted reprojected shoreline；
3. 分别计算 observed→reprojected 与 reprojected→observed 的最近边界距离；
4. 拼接两个方向的距离，计算 mean、P50 和 P95。

重算 P95 与 23 条 `self_consistency.json` 中的既有 P95 完全一致，最大绝对差为 0 px，说明本诊断没有改变正式指标定义。

### 3.2 方向性诊断

现有边界距离采用最近邻匹配，边界点数不相等，并且最近邻关系可能多对一。它不是物理上一一对应的岸线点，因此：

```text
signed_dx_mean_px = unavailable
signed_dy_mean_px = unavailable
signed_dx_median_px = unavailable
signed_dy_median_px = unavailable
```

没有为了得到方向而伪造点对应。

作为独立的探索性辅助指标，诊断在 Camera known region 内对 observed/reprojected water masks 使用 phase correlation，估计“需要施加到 reprojected mask 才能与 observed mask 对齐”的全局平移。该指标用于判断是否存在固定方向趋势，不进入 quality gate，也不等价于点级对应。

### 3.3 Unknown-aware 域对照

为诊断旧 3 px 阈值是否与 unknown-aware 域不匹配，在不改变正式 gate 的前提下，用同一对 masks 额外计算了忽略 unknown 的全图 P95。该结果只作为反事实对照，正式评价域仍为 Camera known region。

## 4. P95 总体分布

| 指标 | 结果 |
|---|---:|
| sequence 数 | 23 |
| 最小值 | 3.1623 px |
| 中位数 | 5.6830 px |
| 均值 | 9.0704 px |
| 最大值 | 37.8284 px |
| 3–4 px（>3 且 ≤4） | 5 |
| >5 px | 15 |
| >8 px | 7 |
| >10 px | 5 |

分布明显右偏：多数序列不是只比 3 px 略高，少量 30 px 以上极端值显著拉高均值。单纯把阈值从 3 px 小幅提高不能解决全部问题。

## 5. 按水深分析

| 水深 | n | P95 最小 | P95 中位 | P95 均值 | P95 最大 | Camera IoU 均值 | 水位误差均值 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 cm | 5 | 3.6056 | 5.6569 | 5.9319 | 9.6599 | 0.5597 | 0.0212 m |
| 10 cm | 7 | 3.1623 | 5.0990 | 5.9500 | 14.2038 | 0.7262 | 0.0227 m |
| 20 cm | 5 | 4.0000 | 9.0000 | 9.2640 | 14.8795 | 0.7418 | 0.0423 m |
| 40 cm | 6 | 5.0000 | 6.0778 | 15.1651 | 37.8284 | 0.8140 | 0.0377 m |

P95 与名义水深的 Pearson 相关为 0.446，属于中等正相关，但 40 cm 均值主要受 37.83 和 31.01 px 两个极端序列影响，其中位数仅 6.08 px。因此不能把问题简化为“水越深，固定投影偏移越大”。

P95 与水位绝对误差的相关为 0.596，说明水位估计和由水位生成的 DEM 水域形状对重投影边界有实际影响，支持“重建/水位误差是贡献因素”，但不是唯一原因。

## 6. 按雨强分析

| 雨强 | n | P95 最小 | P95 中位 | P95 均值 | P95 最大 | Camera IoU | Camera reprojection IoU | 水位误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| light | 6 | 3.6056 | 5.6699 | 10.9810 | 31.0064 | 0.4721 | 0.6548 | 0.0630 m |
| moderate | 9 | 3.1623 | 5.0000 | 5.9903 | 13.4406 | 0.7917 | 0.8621 | 0.0214 m |
| heavy | 8 | 4.0000 | 6.1493 | 11.1027 | 37.8284 | 0.8147 | 0.8612 | 0.0165 m |

light 的主要问题首先来自视觉证据：Camera IoU 最低、unknown fraction 均值最高（0.946）、水位和 depth 误差最大。其 P95 中位数 5.67 px 与全体中位数接近，但 31.01 px 的极端值拉高均值，说明视觉岸线错误会进一步放大几何重投影尾部误差。

heavy 的平均 P95 同样被 40 cm/heavy/204 的 37.83 px 极端值主导，不能据此认为 heavy 普遍比 moderate 存在更大的固定标定偏差。

## 7. 方向性偏移

23 条的全局 phase-correlation 平移中位数为：

- dx = -0.528 px；
- dy = -0.291 px；
- response 中位数 = 0.322。

以 response ≥ 0.20 仅作为可报告级别时，共 16 条：

- dx 中位数 = -1.164 px；
- dy 中位数 = -0.063 px；
- dx 为负占 75%，为正占 25%；
- dy 正负各占 50%。

按雨强分组，reportable dx 中位数从 heavy 的 +2.688 px 到 light 的 -1.472 px、moderate 的 -2.137 px；按水深也没有一致的 dx/dy。高 Camera IoU 的 8 条中同样同时出现正负 dx。

结论：没有证据支持一个稳定、单向、可用固定平移补偿消除的系统偏移。轻微负 dx 倾向存在，但符号和幅度不够一致，不能据此修改外参或平移 Camera/DEM 结果。

## 8. Unknown-aware 评价域检查

尽管 unknown fraction 很高（约 0.81–0.98），正式 unknown-aware P95 与反事实全图 P95 在 23 条序列中逐条完全一致：

- 两者中位数均为 5.6830 px；
- delta 均值和中位数均为 0 px；
- 23/23 差值为 0；
- 两种域下均为 23/23 超过 3 px；
- P95 与 unknown fraction 的相关仅为 -0.152。

因此，当前 artifacts 不支持“unknown-aware 评价域与旧 3 px 阈值不匹配”是本轮全体 reject 的原因。这里的全图计算只是诊断对照，没有更改正式 known-region 评价域。

## 9. Ray–DEM 数值求交

已有 shoreline intersection 点显示：

- 23 条 minimum intersection success rate = 1.0；
- 各序列 absolute residual P95 的最大值约为 `1.04e-8 m`；
- 全部交点 absolute residual 最大值约为 `1.22e-8 m`。

这排除了 ray 与 DEM 求交算法的数值收敛误差作为像素级 P95 的主因。该检查不等于独立相机内外参标定验证，因此不能完全排除标定参数错误；但若固定标定错误是主因，通常应出现跨 seed 更一致的方向和幅度，而本次未观察到。

## 10. Camera IoU ≥ 0.8 但 geometry reject 的 8 条

| Case | 雨强/seed | Camera IoU | Boundary P50 | Boundary P95 | Reprojection IoU | 水位误差 | 主要 geometry 原因 |
|---|---|---:|---:|---:|---:|---:|---|
| 5 cm | moderate/42 | 0.8911 | 1.000 | 3.606 | 0.8986 | 0.0042 m | P95；reprojection IoU |
| 10 cm | moderate/42 | 0.8874 | 1.000 | 3.606 | 0.9140 | 0.0069 m | P95 |
| 20 cm | heavy/43 | 0.8748 | 1.000 | 4.000 | 0.9375 | 0.0245 m | P95 |
| 20 cm | moderate/42 | 0.8934 | 1.414 | 5.000 | 0.8999 | 0.0067 m | P95；reprojection IoU |
| 40 cm | heavy/43 | 0.9444 | 1.414 | 5.000 | 0.9456 | 0.0029 m | P95；多/不可观测盆地 |
| 40 cm | heavy/112 | 0.9271 | 2.000 | 6.325 | 0.9258 | 0.0054 m | P95；多/不可观测盆地 |
| 40 cm | heavy/202 | 0.9209 | 2.000 | 5.831 | 0.9351 | 0.0117 m | P95；多/不可观测盆地 |
| 40 cm | moderate/42 | 0.9204 | 1.414 | 5.000 | 0.9398 | 0.0140 m | P95 |

其中 5/10 cm moderate/42 的 P95 仅为 3.606 px，且水位误差较小，是 3 px 门限可能偏严的直接案例；但另外多条达到 5–6.3 px，并且 40 cm 仍存在不可观测盆地语义。不能以这 8 条为由整体放宽阈值并恢复权威测量。

## 11. 极端失败序列

最大 P95 的五条为：

1. 40 cm/heavy/204：37.83 px，time stability 仅 0.008，Camera IoU 0.742；
2. 40 cm/light/41：31.01 px，Camera IoU 0.429，水位误差 0.166 m；
3. 20 cm/light/41：14.88 px，Camera IoU 0.536，水位误差 0.098 m；
4. 10 cm/heavy/202：14.20 px，time stability 0.099；
5. 20 cm/moderate/203：13.44 px，Camera IoU 0.673。

P95 与 water-mask time stability 的相关为 -0.646，与 Camera reprojection IoU 的相关为 -0.622。极端尾部明显随时域岸线不稳定和重投影区域不一致增加，支持视觉岸线/水位重建问题，而不是单一固定方向偏移。

## 12. 对候选原因 A–F 的判断

| 假设 | 当前判断 | 证据 |
|---|---|---|
| A. 固定方向系统偏移 | 不支持为主因 | phase dx/dy 符号跨 sequence、雨强和水深不一致；dy 正负各半 |
| B. 内外参或坐标变换错误 | 未完全排除，但不支持为主因 | 同一标定下 P95 从 3.16 到 37.83 px；无稳定方向；ray–DEM 数值残差约 1e-8 m |
| C. DEM 边界重建产生偏移 | 中等支持 | P95 与水位误差相关 0.596；高 Camera IoU 下仍有 3.6–6.3 px 边界差异 |
| D. Camera mask 岸线不稳定 | 对极端尾部支持最强 | P95 与 time stability 相关 -0.646；最大 P95 序列稳定性极低 |
| E. unknown-aware 域与旧阈值不匹配 | 当前数据不支持 | known/full P95 在 23 条上完全相同；与 unknown fraction 相关很弱 |
| F. 3 px 阈值过严 | 对近阈值案例有部分证据，但不足以修改 | 5 条位于 3–4 px；同时 15 条 >5、7 条 >8、5 条 >10 |

综合判断：最合理的优先级是 **D（先处理视觉岸线稳定性与极端序列）→ C（再检查水位/低地重建与重投影边界形状）→ B（用独立标定靶点做只读验证）→ 最后才重新标定 F（阈值分布）**。A 和 E 当前没有足够支持。

## 13. 是否支持修改 3 px 阈值

目前不支持直接修改。

一方面，5 条序列只略高于阈值，且部分高 Camera IoU 序列的水位误差很小，说明 3 px 对离散岸线可能偏严格。另一方面，大多数序列不是近阈值失败：15 条 >5 px，5 条 >10 px，最高 37.83 px。此时整体放宽阈值会把视觉岸线不稳定、水位错误和不可观测盆地一起放行。

下一步应先：

1. 在不改 prediction/gate 的诊断分支中定位 5 个 >10 px 序列的时域岸线和重投影边界差异；
2. 对同一 Camera/DEM 使用人工选取的静态 map landmark 或已知平面点做独立内外参投影检查；
3. 分离“水位改变导致的边界形变”与“整体刚性平移”；
4. 在冻结算法、独立 calibration set 上统计可接受样本的 boundary P95 分布；
5. 只有确认标定、视觉岸线和重建误差已受控后，才讨论阈值重标定。

## 14. 输出

诊断输出位于：

```text
outputs/synthetic_visual_to_depth_integration/geometry_diagnostics/
```

包含：

- `geometry_diagnostics.csv`
- `geometry_diagnostics.json`
- `summary.json`
- `boundary_p95_by_depth.png`
- `boundary_p95_by_rain_level.png`
- `camera_iou_vs_boundary_p95.png`
- `water_level_error_vs_boundary_p95.png`
- `geometry_reject_reason_matrix.png`

这些自动生成 artifacts 继续由现有 `outputs/synthetic_visual_to_depth_integration/` 忽略规则排除，不进入 Git。
