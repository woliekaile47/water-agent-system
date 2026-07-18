# Phase 2D-C-8-1：多指标 Quality Gate 重构协议冻结

## 1. 阶段目标

本阶段只冻结新版 quality gate 的目标、证据分区、结果语义和验证协议，不修改现有 prediction、`configs/water_surface_aware_quality_gate.yaml` 或任何运行时阈值。

项目当前首要精度目标是水位绝对误差不超过 3 cm。3 px 是 Camera mask 与 DEM 重投影边界之间的图像距离阈值，两者量纲不同，不能直接换算，也不能把 3 px 单独当成水位精度判据。

## 2. 现有证据审计

### C5：人工提示多样本

- 36/36 样本水位误差不超过 3 cm；
- 原 gate 为 19 pass、17 reject；
- 3/4/5 px 离线对照曾提示 5 px 可能减少误拒，但该批数据已参与候选提出，不能再作为最终确认集；
- 人工提示结果不能证明自动积水识别完成。

### C6：自动提示单帧

- seed 301：8/12 个 SAM 2 mask 达到既有 Camera 离线研究条件；
- 固定规则 seed 302：仍为 8/12；
- 5 cm heavy 与 10 cm heavy 稳定复现明显过分割；
- Prompt 状态、SAM 2 model score 和最终语义准确率并非一一对应；
- 说明 gate 必须分离提示可信度、Camera mask 语义质量和几何质量。

### C7：自动提示视频传播与连续几何

- 123 帧中 67 帧达到 Camera mask 离线研究条件；
- 120/123 帧水位误差不超过 3 cm；
- prediction-side gate 拒绝 91 帧，其中 88 帧仍满足 3 cm 水位目标；
- 20 cm moderate 的 10 个 boundary-only reject 全部满足 3 cm，且面积/体积误差很小；
- 5 cm heavy 虽多数水位误差达标，但面积和体积不稳定，不能仅按水位放行；
- 40 cm light 的 Camera 可见主 basin 估计准确，但第二 basin 完全不可观测，全局面积只能判定不完整。

## 3. 新结果语义

新版 gate 必须分别输出两个作用域：

1. `camera_visible_estimate`：只对 Camera 有证据的主水域负责。即使全局存在不可观测 basin，只要语义、几何、时序和硬安全条件均满足，可见结果仍可保持研究有效；
2. `global_scene_estimate`：只有不存在不可观测或无法区分的 candidate basin 时才允许标记完整。

存在不可观测 basin 时，不得用 Ground DEM 低洼先验自动补水，也不得把 Camera 可见面积冒充道路全局面积。

## 4. 分层门控结构

### A. 硬安全层

文件缺失、NaN/Inf、负水深、物理最大深度超限、水位不收敛、seed 无效或无 selected basin，必须拒绝全部作用域。

### B. Camera 语义层

综合自动提示状态、Camera 重投影 IoU、边缘接触、主体连通域比例、碎片/孔洞诊断以及浅水强雨风险。SAM 2 model score 不能单独作为积水语义判据。

### C. 岸线几何层

继续使用 ray 成功率、有效岸线样本数、岸线高程 MAD/IQR 和主体外岸线重投影误差。Boundary P95 保留为组合诊断量，不再设计为唯一否决依据。

### D. 时序稳定层

使用相邻水位变化 P95、水位窗口标准差、面积/体积 CV 和相邻 mask IoU。时序稳定只能证明结果没有剧烈漂移，不能证明语义正确。

### E. 可观测性层

不可观测或歧义 basin 必须拒绝 `global_scene_estimate`；若其他条件通过，可保留 `camera_visible_estimate`，并明确面积/体积仅代表 Camera 可见范围。

## 5. GT 隔离与评价目标

Prediction-side gate 禁止读取 Camera/DEM mask GT、真实水位、depth、area、volume、nominal depth 或任何 evaluation 状态。GT 只由独立 evaluation 读取。

本轮只冻结已经明确的 3 cm 水位目标。面积和体积的正式相对误差上限尚未由项目需求批准，因此暂不编造阈值；它们继续作为重要的二级评价指标和浅水风险证据。

## 6. 证据分区

现有 C5、C6 和 C7 全部归入 development evidence，因为它们已经参与问题发现、固定规则或门控设计。不能再用这些数据同时完成最终确认。

最终确认矩阵冻结为建议目标：4 个水深 × 3 个雨强 × 新 seed 303 × 预先固定 frame，共至少 12 个样本。当前仓库尚无完整 seed 303 矩阵，因此状态为 `data_not_yet_frozen`。阈值必须在生成或查看该确认矩阵结果前冻结；确认结果不得反向用于选择阈值。

## 7. 当前结论

C8-1 已确定“分层门控 + 双结果作用域 + 独立确认”的路线。现有 3 px 运行时阈值保持不变，所有结果继续 `authoritative=false`、`eligible_for_downstream=false`。

下一小阶段为 C8-2：只使用 development evidence 建立离线候选 gate，报告旧/新 gate 的误拒与不安全放行对照，冻结候选参数；不覆盖旧 gate。之后 C8-3 生成并冻结 seed 303 独立矩阵，完成一次性确认。
