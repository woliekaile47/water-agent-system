# Phase 2D-A 全合成视觉到水深集成实验

## 1. 研究目标

本实验在不读取 Ground Truth（GT）的预测链路中，将 Phase 2C 的连续 RGB 时域积水识别结果接入 Phase 2B 的 Camera-mask-to-DEM 几何反演，验证以下最小闭环：

```text
synthetic RGB frames
→ temporal camera water / unknown masks
→ trusted shoreline
→ ray–ground-DEM intersection
→ shoreline water-level estimation
→ connected lowland reconstruction
→ depth / area / volume candidate
→ visual + geometry + integration quality gates
→ independent post-prediction evaluation
```

本阶段的目标是建立可重复、可审计的离线集成与独立评价基线，不是证明系统已具备真实道路部署能力，也不接入 S5–S8。

## 2. 数据集组成

批量扫描 `data/simulation_dynamic/` 下全部有效序列，共 33 个：

- dry：10 个；
- water：23 个；
- 名义水深：5、10、20、40 cm；
- 雨强：light、moderate、heavy；
- 每个组合包含配置中已有的确定性 seed。

动态 Camera mask GT 来自各序列的 `ground_truth/water_mask.png`；DEM mask、depth、真实 water level、area 和 volume 来自对应静态 case 的 `data/simulation/<case_id>/ground_truth/` 与 manifest。dry 序列只评价误报，不构造不存在的 water-level、depth、area 或 volume GT。

这些数据属于 **synthetic visual abstraction**：雨滴、飞溅和涟漪是可重复的视觉抽象，**not real fluid simulation**，并且 **not real-world validated**。

## 3. Prediction / GT 隔离

批处理对每个序列严格按顺序执行：

1. 仅用 RGB frames、配置、dry Ground DEM 和传感器标定完成 prediction；
2. 写完 prediction manifest、Camera water/unknown masks、三个 quality gate 及候选几何输出；
3. 独立 evaluation 模块首先验证 prediction 输出完整，并检查 manifest 中 `data_role=prediction` 及 `ground_truth_or_metadata_read_during_prediction=false`；
4. 只有上述检查通过，evaluation 才读取动态与静态 GT；
5. 写入单序列 `evaluation_metrics.json`，最后生成全数据集汇总。

GT loader 只出现在 `src/evaluation/`。`scripts/run_synthetic_visual_to_depth_integration.py` 和 `src/integration/` 不导入 GT loader。相关测试覆盖 prediction 未完成时禁止读取 GT，以及 prediction 源码无 GT loader 导入。

## 4. 算法链路与 unknown 语义

预测使用既有 Phase 2C-2A 规则 mask 作为几何输入；学习分类器仅用于诊断，不参与本阶段几何反演。Camera water mask 与 unknown mask 必须互斥。

unknown 的统一语义是：

```text
no_temporal_evidence_not_confirmed_dry
```

即“缺少足够时域证据”，不能当作已确认干地。几何适配层因此：

- 只在 water 与已知非水区域的界面建立 trusted shoreline；
- 排除 water–unknown 界面，避免把未知边界错误解释成零水深岸线；
- 重建候选低地时区分相机已知覆盖、unknown-only 和不可观测盆地；
- 重投影一致性指标只在 Camera known region 计算。

几何链路复用冻结的 Phase 2B 算法，通过 unknown-aware adapter 完成岸线求交、稳健水位估计、低地连通重建、深度/面积/体积计算及 Camera 重投影一致性诊断。

## 5. 联合 quality gate

联合门控同时继承 visual gate 与 geometry gate：

- 任一 gate 为 `reject`，integration gate 为 `reject`；
- 任一 gate 为 `partial`，或全局几何状态不是 `complete`，integration 最多为 `partial`；
- 只有 visual 与 geometry 均为 `pass` 且全局状态为 `complete`，integration 才能为 `pass`；
- reject 后即使仍有候选数值，也只能标记为 `rejected_candidate`，`authoritative_measurement_available=false`；
- 本实验固定 `eligible_for_downstream=false`。

候选结果的评价角色为 `rejected_candidate_diagnostic`，不得计入权威测量成功率。

## 6. 评价指标

- Camera mask：whole-image IoU、known-region IoU、precision、recall、F1、unknown fraction；
- water level：绝对误差；
- DEM mask：IoU、precision、recall、1-cell boundary F1；
- depth：GT water region 与 prediction/GT union 上的 MAE、RMSE，以及最大/平均水深绝对误差；
- area / volume：预测值、GT、绝对误差、相对误差；
- dry：false-water pixels、fraction、components、false-positive area；
- gate：visual、geometry、integration、measurement status、权威性、全局/可观测语义及 reject reasons；
- 40 cm：按 GT DEM 连通组件分别记录 cell 数、预测交集、recall、Camera 可观测性和组件角色。

缺失或 unavailable 值使用 `null`，从均值中排除，绝不替换成 0。dry 场景的零 Camera IoU/误报是实际的零误报结果，不是缺失值填充。

## 7. 完整数据集结果

批处理选择 33 个序列，成功评价 33 个，失败 0 个。CSV 为表头加 33 行，与 JSON sequence 数一致。

| 项目 | 结果 |
|---|---:|
| visual gate pass | 17 |
| visual gate partial | 16 |
| geometry gate reject | 33 |
| integration gate reject | 33 |
| measurement unavailable（dry） | 10 |
| measurement rejected_candidate（water） | 23 |
| authoritative measurement | 0 |
| water 序列权威测量成功率 | 0% |
| eligible_for_downstream | false |

因此，本实验没有任何可作为正式水深测量的输出。以下 water 指标全部是被拒候选的离线诊断指标。

## 8. 按名义水深分析

| 水深 | 序列数 | Camera IoU | Known IoU | 水位绝对误差 (m) | DEM IoU | Depth MAE (m) | Area 相对误差 | Volume 相对误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 cm | 5 | 0.5597 | 0.5660 | 0.0212 | 0.5948 | 0.0167 | 43.67% | 70.84% |
| 10 cm | 7 | 0.7262 | 0.7294 | 0.0227 | 0.7715 | 0.0200 | 23.25% | 40.92% |
| 20 cm | 5 | 0.7418 | 0.7465 | 0.0423 | 0.7648 | 0.0350 | 23.87% | 37.20% |
| 40 cm | 6 | 0.8140 | 0.8162 | 0.0377 | 0.7224 | 0.0249 | 27.22% | 17.63% |

5 cm 的视觉 IoU、面积和体积相对误差最差，说明浅水视觉证据和几何量化仍较脆弱。40 cm 的 Camera IoU 最高，但全局 DEM IoU 并未同步提高，主要受不可观测第二盆地影响。

## 9. 按雨强分析

| 雨强 | 序列数 | Camera IoU | Known IoU | 水位绝对误差 (m) | DEM IoU | Depth MAE (m) | Area 相对误差 | Volume 相对误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| light | 6 | 0.4721 | 0.4852 | 0.0630 | 0.4803 | 0.0424 | 51.97% | 71.60% |
| moderate | 9 | 0.7917 | 0.7930 | 0.0214 | 0.8127 | 0.0188 | 18.83% | 28.86% |
| heavy | 8 | 0.8147 | 0.8147 | 0.0165 | 0.7922 | 0.0154 | 22.81% | 30.38% |

light 明显弱于 moderate/heavy：Camera IoU 更低，水位、depth、area 和 volume 误差均更高。这符合合成视觉事件稀疏时，时域规则难以形成稳定水域证据的现象，但不能外推为真实降雨规律。

## 10. Dry 误报分析

10 个 dry 序列的结果为：

- false-water pixels 总数：0；
- mean false-water fraction：0；
- false-water components 总数：0；
- mean false-positive area：0 m²。

dry 上未产生水位、depth、area 或 volume 评价字段。该结果仅证明当前合成 dry 数据上无误报，不代表真实湿润路面、反光或夜间场景的误报率。

## 11. 40 cm 不可观测盆地

40 cm GT DEM mask 始终包含两个独立组件：可见主盆地 1848 cells，以及不可见第二盆地 363 cells。第二盆地在所有 6 个序列中 Camera 投影像素均为 0、prediction intersection 均为 0，作为 `unobservable_secondary_basin` 单独报告，未通过 DEM 低洼区域自动填充。

| 雨强 / seed | 主盆地交集 cells | 主盆地 recall | 第二盆地投影像素 | 第二盆地 recall |
|---|---:|---:|---:|---:|
| heavy / 43 | 1848 | 1.0000 | 0 | 0.0000 |
| heavy / 112 | 1800 | 0.9740 | 0 | 0.0000 |
| heavy / 202 | 1749 | 0.9464 | 0 | 0.0000 |
| heavy / 204 | 1632 | 0.8831 | 0 | 0.0000 |
| light / 41 | 852 | 0.4610 | 0 | 0.0000 |
| moderate / 42 | 1735 | 0.9389 | 0 | 0.0000 |

这不是可由单 Camera mask 安全恢复的普通 FN。强制填充第二盆地会把 DEM 低洼先验冒充视觉观测，因此全局结果必须维持不可用/拒绝语义。

## 12. 失败案例与门控观察

最常见 reject 原因是 `geometry:boundary_reprojection_error_above_threshold`，出现 23 次，即全部 23 个 water 候选都超过现有 `max_boundary_reprojection_p95_px=3.0` 门限。其他高频原因包括 Camera 重投影 IoU 低于阈值 17 次、视觉高置信水域轨迹不足 15 次、mask 时间稳定性低 11 次。

存在 8 个 Camera IoU ≥ 0.8 但 geometry reject 的序列，证明“视觉 mask 较好”不等于“几何反演可信”：

- 10 cm moderate/42；
- 20 cm heavy/43、moderate/42；
- 40 cm heavy/43、heavy/112、heavy/202、moderate/42；
- 5 cm moderate/42。

另有 6 个 visual partial 序列仍产生候选几何值，但 geometry 全部 reject，因此没有“visual partial 且几何结果经 gate 证明稳定”的证据，只能保留为候选诊断。

3.0 px 阈值在该数据集上表现为 23/23 water 候选全部触发，构成“可能普遍偏紧或误差定义/标定存在系统偏差”的数据证据；但同一批结果同时存在真实重投影不一致、岸线离散和不可观测性，尚不足以直接判定阈值错误。本批遵守要求，未修改阈值、算法、GT 或评价域。后续应在独立标定集上分析 boundary P95 分布及误差来源后再决定是否调整。

## 13. 局限性与是否适合进入 S5–S8

当前局限包括：

- 合成雨滴/飞溅/涟漪仅是视觉抽象，不是复杂流体仿真；
- 未覆盖真实湿路材质、车灯/路灯、夜间镜面反射、雨幕遮挡、车辆与相机抖动；
- 单 Camera 存在明确不可观测盆地，无法提供全局完整面积/体积；
- 轻雨和浅水证据不足；
- boundary P95 门控在所有 water 候选上触发，需要独立诊断；
- 所有 geometry 与 integration gate 均 reject，权威测量成功率为 0。

结论：本阶段 **eligible_for_downstream=false**，不适合接入正式 S5–S8，也不能作为真实场景性能声明。现有结果只用于算法排查和后续实验对照。

## 14. 下一步建议

1. 保持当前 prediction 核心和阈值冻结，先用独立标定数据分解 boundary P95 的相机标定、离散岸线、投影和水位误差来源；
2. 为 light 与 5 cm 增加不调 case 参数的统一视觉证据改进实验，并继续与本基线盲评对比；
3. 增加 Camera 覆盖或多 Camera，解决 40 cm 第二盆地不可观测问题，不允许仅凭 DEM 自动补全；
4. 扩充湿润路面、照明、遮挡和相机扰动的合成域随机化；
5. 在冻结配置后开展真实视频、小规模人工标注和独立测试集验证；
6. 只有联合 gate 在独立数据上产生可重复的 pass/可信 partial，并完成真实场景验证后，才重新评估 S5–S8 接入。

## 15. 产物

汇总目录为 `outputs/synthetic_visual_to_depth_integration/dataset_evaluation/`，包含：

- `dataset_summary.json`、`dataset_summary.csv`；
- `gate_distribution.json`；
- `metrics_by_depth.json`、`metrics_by_rain_level.json`、`metrics_by_measurement_status.json`；
- `failure_reasons.json`、`batch_failures.json`；
- 六张独立 matplotlib 图：Camera IoU、水位误差、depth MAE、gate 分布、雨强指标和 dry 误报。

自动生成评价 JSON、CSV 和 PNG 均为本地实验输出，不纳入 Git。
