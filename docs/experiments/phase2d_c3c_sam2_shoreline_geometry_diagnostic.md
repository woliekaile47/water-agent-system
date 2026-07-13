# Phase 2D-C-3C：SAM 2 可见水域岸线到 Ground DEM 的预测侧几何诊断

## 实验目的

本实验验证人工提示 SAM 2 单帧可见水域候选能否通过现有 Phase 2B 几何链路形成可审计的离线诊断结果。输入只包括人工提示候选 mask、外岸线、人工正点、相机几何和 dry baseline Ground DEM；没有读取积水状态 Ground Truth。

本结果不是自动水体识别，不是权威积水测量，也不允许进入正式 S5-S8。

## 输入与追溯

- 原始图像：`data/simulation_dynamic/sim_water_5cm_001/heavy/seed_43/frames/frame_000059.png`
- 图像 SHA-256：`06e70868fa7933ffcc2d7ef15032631f71d3f5d1f7b0df2898b9edd3e6f14469`
- SAM 2 selected component mask：`360 × 640`，2,795 pixels
- 完整外岸线：233 个 `[x, y]` 点
- 固定采样外岸线：128 个 `[x, y]` 点
- 人工提示：5 个正点、7 个负点和一个 box
- dry Ground DEM：`70 × 180`，0.1 m resolution，frame=`map`
- Camera：640 × 360，`fx=fy=554.2562584220408`，`cx=319.5`，`cy=179.5`

`source_trace.json` 只作为文件保管副本参与 SHA-256 记录，预测代码不解析它。相机参数直接读取 Phase 2B 已使用的 `simulation/config/sensors.yaml`。

## Phase 2B 复用关系

本阶段直接复用以下实现，没有修改其算法或阈值：

- `camera_ray_map`：Camera pixel ray 到 map；
- `bilinear_dem_height` / `intersect_ray_with_dem`：DEM 双线性采样、步进符号变化检测和二分求交；
- `robust_filter` / `estimate_water_level_from_shoreline`：统一 MAD 过滤和 robust median；
- `reconstruct_connected_lowland`：低地连通域重建；
- `reproject_water_surface` / `camera_reprojection_consistency`：Camera 重投影；
- `evaluate_water_surface_aware_quality_gate`：原 Phase 2B prediction-side gate。

新增适配层仅负责读取 SAM 2 `[x,y]` 岸线、记录逐射线数据、构造人工提示 seed、强制 seed-connected 输出、补充 outer-boundary 指标和生成实验文件。

## 岸线射线与求交

- 总岸线射线：128
- 成功求交：128
- 成功率：1.0
- 失败原因：无
- Camera mask edge-touch ratio：0.0
- ray–DEM 残差由逐射线 JSON/CSV 保存；求交仅针对 dry Ground DEM，没有使用水面平面。

## 岸线 Ground DEM 高程

| 统计量 | 数值（m） |
|---|---:|
| min | -0.409865480 |
| p10 | -0.396516402 |
| p25 | -0.391205018 |
| median | -0.365026741 |
| p75 | -0.324959215 |
| p90 | -0.284290135 |
| max | -0.246491324 |
| mean | -0.353340163 |
| std | 0.043487761 |
| MAD | 0.026947952 |
| IQR | 0.066245803 |

Phase 2B 固定 MAD 配置保留全部 128 个样本，没有判定离群点。稳健中位数水位为：

`estimated_water_level_m = -0.365026740965642`

估计过程收敛，但 MAD 和 IQR 均超过现有 gate 上限，说明这条人工提示外岸线落到 dry DEM 后的地面高程离散较大。

## Camera seed 与低地重建

- Camera seed pixels：37
- 成功 seed rays：37，成功率 1.0
- 去重 DEM seed cells：35
- 位于候选低地内的有效 seed cells：30
- candidate basins：1
- selected basins：1
- predicted DEM water cells：257

所有 seed 都使用与岸线相同的 ray–DEM 求交。输出只包含有 seed 的低地连通域；没有依据 DEM 低洼先验补全无 Camera seed 的独立 basin。

## 诊断性水深、面积和体积

以下结果只具有 `camera_visible_candidate_estimate` 语义：

- mean depth：3.8545 cm
- median depth：3.8794 cm
- max depth：7.8868 cm
- area：2.5700 m²
- volume：0.099060543 m³
- negative depth count：0
- Inf depth count：0

这些数值没有使用积水 GT 校正，不能解释为整段道路的真实积水测量。

## Camera 重投影自一致性

- IoU：0.694469
- precision：0.808562
- recall / Camera coverage ratio：0.831127
- boundary P50：4.0 px
- boundary P95：8.944272 px
- outer-boundary P50：4.123106 px
- outer-boundary P95：9.0 px
- valid projection ratio：1.0
- predicted mask touches image border：false

## Prediction-side quality gate

`quality_status = reject`

拒绝原因：

- `camera_reprojection_iou_below_threshold`
- `boundary_reprojection_error_above_threshold`
- `shoreline_height_mad_above_threshold`
- `shoreline_height_iqr_above_threshold`

`geometry_diagnostic_readiness = diagnostic_only`

这意味着 ray–DEM 与 seed 链路可运行且可审计，但当前人工提示岸线在 Ground DEM 上不满足正式 Phase 2B 自一致性要求。没有修改 0.90 IoU、3 px P95、0.020 m MAD 或 0.060 m IQR 等现有阈值。

## Ground Truth 防泄漏

- prediction CLI 没有 `--gt-mask`、`--gt-water-level` 或 `--known-depth`；
- prediction 模块没有加载积水 mask、积水水位、depth map、面积或体积 GT；
- `source_trace.json` 不由 prediction 解析；
- quality gate 输入不含 GT IoU、recall 或 area error；
- 所有结果均为 `ground_truth_used=false`、`authoritative=false`；
- `eligible_for_formal_s5_s8=false`、`eligible_for_downstream=false`。

## 结论

人工提示 SAM 2 候选已经具备进行单帧、预测侧几何诊断的接口条件，且 128 条岸线射线全部成功与 dry DEM 求交。但当前候选外岸线的 DEM 高程离散和 Camera 重投影误差较大，必须保持 reject / diagnostic-only，不得进入正式预警链路。下一步如继续研究，应优先审计人工提示外岸线与真实可见水面岸线的结构差异，而不是降低现有 gate 阈值或用 Ground Truth 调参。
