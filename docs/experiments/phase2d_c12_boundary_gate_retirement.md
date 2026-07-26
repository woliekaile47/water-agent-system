# Phase 2D-C-12：旧 3 px 边界否决规则退出

## 1. 目标

本阶段移除旧质量门中“边界重投影 P95 超过 3 px 即单独拒绝”的运行时行为，同时保留边界误差作为预测侧自一致性诊断指标。

该改动不修改 Camera mask、ray–DEM、水位估计、DEM 重建、S5–S8 算法或任何 Ground Truth 评价公式。

## 2. 新边界语义

当前统一语义为：

- 优先读取主体外岸线 `outer_boundary_reprojection_p95_px`；
- 若旧结果没有主体外岸线字段，则兼容读取 `boundary_reprojection_p95_px`；
- P95 不超过 5 px 时不产生边界警告；
- P95 超过 5 px 时记录 `outer_boundary_reprojection_p95_above_advisory_threshold`；
- 边界指标缺失或非有限时记录 unavailable 警告；
- 边界指标不再单独改变 pass/reject。

5 px 是诊断警告线，不是新的单项否决线。

## 3. 仍参与拒绝的指标

质量门继续检查：

- 岸线射线成功率和 Camera seed 有效性；
- Camera 重投影 IoU；
- 有效岸线样本数及 MAD/IQR；
- 水位是否收敛；
- NaN、Inf、负水深和物理最大深度；
- 多盆地歧义和 Camera 不可观测区域；
- C8 候选门中的水位、面积和体积时序稳定性。

因此退出 3 px 否决不会使质量门失去对错误结果的保护。

## 4. 兼容策略

- `legacy_runtime_gate` 的历史状态和拒绝原因继续保留用于审计；
- C9 统一状态中的 active gate 改为 `phase2d_c8_candidate_gate_v1`；
- legacy gate 标记为 `audit_only`；
- 历史实验文档中的 3 px 结果不回写、不删除；
- 新输出继续保留边界 P95 数值和 advisory warning。

## 5. 安全边界

本阶段仍保持：

- `authoritative=false`；
- 正式真实预警未启用；
- 不使用 Ground Truth 参与 prediction-side gate；
- Camera 不可观测盆地不自动补全；
- 仿真结果不代替真实道路验证。

## 6. 结论

旧 3 px 规则退出的是单项否决权，而不是删除边界诊断。运行链路使用多指标质量门；5 px 仅作为边界异常提醒。
