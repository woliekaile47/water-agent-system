# Phase 2D-C-8-2：多指标候选 Gate 离线对照

## 1. 目标与限制

本阶段在独立模块中实现 Phase 2D-C-8 研究候选 gate，并只读取 C7 已冻结的 123 帧 prediction-side 几何指标进行离线对照。

- 未重新运行 SAM 2、岸线提取、ray–DEM、水位估计或 DEM 重建；
- 未修改旧 gate、`configs/water_surface_aware_quality_gate.yaml` 或 3 px 阈值；
- 候选决策全部完成后，脚本才读取独立 evaluation 文件统计误差；
- GT 不进入候选 gate API；
- 候选结果不替换运行时 gate，保持 `authoritative=false`、`eligible_for_downstream=false`。

## 2. 候选规则

### 帧级硬安全与几何

继续检查几何是否可用、selected basin、有限水位、非负且不超物理上限的水深、ray 成功率、有效岸线样本数以及岸线高程 MAD/IQR。

Camera 重投影 IoU 下限暂保持 0.90。主体外岸线 P95 超过 5 px 仅产生 advisory warning，不再单独拒绝结果；像素误差不得直接换算成水位厘米误差。

### 序列级稳定性

本开发候选使用以下固定条件：

- 水位窗口标准差 ≤ 0.50 cm；
- 相邻水位变化 P95 ≤ 1.00 cm；
- 面积 CV ≤ 10%；
- 体积 CV ≤ 20%。

这些是基于 development evidence 的研究候选，不是最终生产阈值。

### 双作用域语义

- 通过帧级和序列级检查后，Camera 可见结果标记 `camera_visible_status=pass`；
- 存在不可观测或歧义 basin 时，可见结果可以保留，但 `global_scene_status=partial`；
- 不存在这些 basin 时才允许 `global_scene_status=complete`；
- 可见结果本身失败时，全局结果为 `unavailable`。

## 3. 123 帧对照结果

| 场景 | 旧 gate pass/reject | 旧 observable valid | 候选 Camera-visible pass/reject | 候选 global 状态 | 水位 ≤3 cm |
|---|---:|---:|---:|---|---:|
| 5 cm heavy | 1 / 40 | 1 | 0 / 41 | unavailable 41 | 38/41 |
| 20 cm moderate | 31 / 10 | 31 | 41 / 0 | complete 41 | 41/41 |
| 40 cm light | 0 / 41 | 17 | 41 / 0 | partial 41 | 41/41 |
| 合计 | 32 / 91 | 49 | 82 / 41 | complete 41、partial 41、unavailable 41 | 120/123 |

与旧 observable-region 判断相比：

- 34 帧由旧 visible reject 转为候选 visible pass：20 cm 的 10 帧和 40 cm 的 24 帧；
- 1 帧由旧 visible pass 转为候选 reject，属于 5 cm heavy；
- 82 个候选 visible pass 中，水位误差超过 3 cm 的数量为 0；
- 41 个候选 visible reject 中，38 个水位误差仍低于 3 cm，但均属于 5 cm heavy 风险序列。

后者不能简单称为水位目标“误拒”：该序列面积 CV 为 15.23%、体积 CV 为 31.32%，Camera 重投影 IoU 也有 40/41 帧低于 0.90。错误视觉范围仍可能生成表面上接近真实值的水位标量。

## 4. Boundary P95 的新角色

共有 32 帧的主体外岸线 P95 高于 5 px，候选 gate 将其记录为 warning，而不是单独否决：

- 20 cm 的 boundary-only 旧 reject 可在其他指标稳定时恢复为 complete；
- 40 cm 的局部边界尾部不再抹去 Camera 可见主盆地结果，但全局仍因第二 basin 不可观测而保持 partial；
- 5 cm 不依赖 boundary warning 拒绝，而由 Camera 重投影和序列不稳定性拒绝。

这比直接把旧 3 px 改成 5 px 更符合现有证据：边界仍保留诊断作用，同时不再承担唯一决策责任。

## 5. 安全性与证据边界

候选 gate API 只接受 `frame_metrics`、`sequence_metrics` 和配置，不接受 GT。分析脚本在全部 123 帧候选决策完成后才读取冻结 evaluation，用于离线统计 3 cm 目标。

本结果仍不能证明候选 gate 已通过独立验证，因为 C7 三个序列参与了规则设计。尤其当前只有三个代表性序列，尚不足以覆盖所有水深、雨强、seed、dry false-water 和真实场景域差异。

## 6. 结论与下一步

C8-2 证明分层 gate 可以同时做到：

1. 保留浅水强雨风险拒绝；
2. 消除 20 cm 的纯 boundary 保守误拒；
3. 保存 40 cm Camera 可见估计，同时拒绝将其冒充全局完整结果；
4. 不把 boundary P95 或水位误差单独作为全部质量的替代指标。

下一阶段 C8-3 必须在查看结果前冻结候选配置，并使用未参与设计的 seed 303 矩阵进行一次性独立确认。还应加入 dry 序列验证空 mask 不会形成 false-water measurement。在 C8-3 通过前，不覆盖旧 gate，不接入 S5-S8。
