# Phase 2D-C-7-4：连续帧几何结果独立 Ground Truth 评价

## 1. 实验目的

本阶段对 Phase 2D-C-7-3 已冻结的 123 帧 prediction-side 几何标量结果进行独立 Ground Truth（GT）评价，重点验证连续帧水位、面积和体积误差，并分析现有 quality gate 与“水位误差不超过 3 cm”业务目标之间的关系。

本阶段不重新运行 SAM 2，不重新计算岸线、ray–DEM 求交、水位或 DEM 重建，不修改人工/自动提示，不修改任何 gate 阈值，也不将 rejected candidate 转换为 authoritative measurement。所有结果均为离线研究评价，`authoritative=false`、`eligible_for_downstream=false`。

## 2. 冻结与防泄漏协议

评价对象为以下三个固定序列，每个序列 41 帧，共 123 帧：

- 5 cm / heavy / seed 302 / frame 79–119；
- 20 cm / moderate / seed 302 / frame 79–119；
- 40 cm / light / seed 302 / frame 79–119。

程序在第一次读取 GT 前，先验证 C7-3 的逐帧摘要、逐帧详情和序列汇总文件及其 SHA-256。只有全部冻结文件验证通过后，evaluation 模块才读取 GT。GT 不进入 prediction 模块，不参与候选、岸线、水位、seed 或 basin 选择。

由于 C7-3 只为全部帧冻结了标量结果，而没有为每帧冻结 predicted DEM mask 和 depth raster，本阶段不重新运行 prediction 来补齐栅格。因此逐栅格 DEM mask IoU、depth MAE/RMSE/bias 明确标记为 `unavailable`，避免用重算结果冒充冻结结果。

## 3. Ground Truth 一致性

三个 case 的 Camera GT 均为 640×360，sequence GT 与静态 case GT 一致。Ground DEM、DEM water mask 和 depth map shape 均为 70×180；depth 非负且 mask 外为 0；area 与 mask cell count、volume 与 depth 积分数值一致。

40 cm GT 包含两个独立 basin：

- Camera 可见主 basin：1,848 cells，18.48 m²，3.080219001 m³；
- Camera 不可观测第二 basin：363 cells，3.63 m²，0.015617188 m³，Camera 投影像素为 0；
- 全局场景：2,211 cells，22.11 m²，3.095836190 m³。

因此 40 cm 同时报告 `camera_visible_estimate` 与 `global_scene_estimate`，不把不可观测 basin 自动加入 Camera prediction。

## 4. 评价结果

下表均为 41 帧统计。相对误差使用绝对相对误差；表中面积和体积为中位数相对误差。

| 场景 | 水位绝对误差中位数 / 均值 / 最大值 (cm) | ≤3 cm | 可见面积误差 | 全局面积误差 | 可见体积误差 | 全局体积误差 | gate pass / reject |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 cm heavy | 0.6746 / 0.9693 / 3.1235 | 38/41 | 14.56% | 14.56% | 28.78% | 28.78% | 1 / 40 |
| 20 cm moderate | 0.1514 / 0.1769 / 0.4547 | 41/41 | 0.42% | 0.42% | 1.60% | 1.60% | 31 / 10 |
| 40 cm light | 0.6292 / 0.6580 / 1.0980 | 41/41 | 2.81% | 18.77% | 3.72% | 4.20% | 0 / 41 |

总体上，120/123 帧满足水位误差不超过 3 cm。现有 gate 共拒绝 91 帧，其中 88 帧仍满足 3 cm 水位目标。

## 5. 分场景解释

### 5 cm heavy

水位误差多数低于 3 cm，但面积和体积误差仍明显，体积误差中位数为 28.78%，最大达到 166.09%。这说明浅水条件下视觉边界与水域范围仍不稳定；即使水位标量偶然满足 3 cm，结果也不能仅凭该指标进入下游。

### 20 cm moderate

41 帧全部满足 3 cm 水位目标，面积和体积误差也较低，但其中 10 帧仍因现有边界条件被 reject。这是当前 3 px boundary gate 存在保守误拒的最直接证据。

### 40 cm light

Camera 可见主水域的面积误差中位数仅 2.81%，体积误差中位数 3.72%，水位误差最大 1.10 cm；但由于第二 basin 完全不可观测，全局面积仍低估约 18.77%。41 帧全部被 gate 拒绝，与 `ambiguous_candidate_basin`、`candidate_basin_outside_camera_coverage` 及部分帧的边界原因有关。这个 reject 对“全局场景完整性”是合理的，但不应抹去可见主水域估计的研究价值。

## 6. 对 3 px gate 的判断

目前证据支持“现有 3 px gate 与 3 cm 水位业务目标不等价，并会产生保守误拒”，但不支持直接把 3 px 阈值放宽或删除：

1. 像素边界误差受透视、局部岸线形状和图像位置影响，不能用固定比例直接换算成厘米；
2. 20 cm 的 10 个 reject 表明 pixel-only gate 过严；
3. 5 cm 又表明只看水位误差会放过面积/体积不可靠的浅水结果；
4. 40 cm 需要区分 Camera 可见估计与全局完整估计，不能用同一个二值 gate 表达两种语义。

因此下一阶段应基于多指标和结果语义重新设计 gate，而不是单独调大 boundary P95 阈值。

## 7. 结论与下一步

Phase 2D-C-7 的连续帧传播、Camera GT 评价、prediction-side 几何稳定性和独立几何 GT 评价已经形成闭环。下一步进入 Phase 2D-C-8：

- 将业务目标明确为水位误差、面积/体积可靠性和时序稳定性的组合条件；
- 分离 `camera_visible_estimate` 与 `global_scene_estimate`；
- 保留边界重投影指标作为诊断量，但避免其单独决定所有结果；
- 纳入 shoreline MAD/IQR、ray 成功率、时序水位波动、Camera 可观测覆盖和 candidate basin 语义；
- 使用 held-out 证据冻结新 gate，再做独立验证；
- 在通过该阶段前，不接入正式 S5-S8、Agent、数据库或 Dashboard。
