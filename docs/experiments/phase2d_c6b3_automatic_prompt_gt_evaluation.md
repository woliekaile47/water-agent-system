# Phase 2D-C-6B-3：自动提示 SAM 2 冻结掩膜独立 GT 评价

## 实验目的

本阶段对 Phase 2D-C-6B-2 已冻结的 12 份自动提示 SAM 2 Candidate 1 原始掩膜进行独立 Camera Ground Truth 评价，用于判断“时序视觉证据自动生成提示 → SAM 2 单帧分割”在不同水深和雨强下的分割能力。

## 严格隔离协议

- 评价前一次性验证 12 份 RGB、SAM 2 raw mask 和 SAM 2 summary 的冻结 SHA-256；任一不一致即在读取 GT 前停止。
- 不重新运行时序提示生成、SAM 2、岸线提取、ray–DEM 或水深 prediction。
- GT 只由 `src/evaluation/evaluate_temporal_sam2_mask_gt.py` 读取。
- 本阶段只读取 Camera water mask GT，不读取 water level、DEM water mask、depth map、area 或 volume GT。
- GT 不用于修改提示、选择 Candidate、调参或改变 prediction-side gate。
- 自动提示 SAM 2 输出仍是研究候选，不是 authoritative measurement，不能进入 S5-S8。

## 样本矩阵

固定矩阵为 4 个水深（5/10/20/40 cm）× 3 个雨强（light/moderate/heavy）× seed 301 × frame 149，共 12 个样本。该矩阵与 Phase 2D-C-6B-2 完全一致，不进行逐 case 选择。

## 评价指标

- Camera mask：IoU、precision、recall、F1、FP/FN 像素。
- 外岸线：对称 P50/P95，以及 prediction→GT、GT→prediction 的方向性 P95。
- 拓扑：连通域数量、内部孔洞数量。
- 仅作离线研究统计的既有条件：IoU ≥ 0.90、recall ≥ 0.90、outer boundary P95 ≤ 5 px。该条件不是 prediction-side gate，本阶段不修改任何 gate 阈值。

## 实验结果

12 份冻结输入均在读取 GT 前通过 SHA-256 核验，SAM 2 和 prediction 重跑次数均为 0。8/12 个样本同时满足既有离线研究条件。

总体指标如下：

| 指标 | 最小值 | 中位数 | 均值 | 最大值 |
|---|---:|---:|---:|---:|
| Camera IoU | 0.4824 | 0.9395 | 0.8666 | 0.9732 |
| Precision | 0.4830 | 0.9775 | 0.9112 | 0.9996 |
| Recall | 0.7232 | 0.9801 | 0.9543 | 0.9986 |
| F1 | 0.6509 | 0.9688 | 0.9211 | 0.9864 |
| Outer boundary P95 / px | 2.2361 | 3.3839 | 11.8687 | 69.0062 |

逐样本结果：

| 水深 | 雨强 | Prompt 状态 | IoU | Precision | Recall | F1 | Outer P95 / px | 既有离线条件 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 5 cm | light | diagnostic_only | 0.9321 | 0.9712 | 0.9585 | 0.9648 | 2.7692 | 通过 |
| 5 cm | moderate | pass | 0.8696 | 0.8847 | 0.9807 | 0.9302 | 3.6056 | 未通过（IoU） |
| 5 cm | heavy | pass | 0.4824 | 0.4830 | 0.9977 | 0.6509 | 27.7822 | 未通过 |
| 10 cm | light | diagnostic_only | 0.9397 | 0.9942 | 0.9449 | 0.9689 | 3.0000 | 通过 |
| 10 cm | moderate | pass | 0.9540 | 0.9715 | 0.9815 | 0.9765 | 2.2361 | 通过 |
| 10 cm | heavy | pass | 0.7113 | 0.7120 | 0.9986 | 0.8313 | 18.1927 | 未通过 |
| 20 cm | light | diagnostic_only | 0.9447 | 0.9973 | 0.9471 | 0.9716 | 4.0000 | 通过 |
| 20 cm | moderate | pass | 0.9571 | 0.9929 | 0.9637 | 0.9781 | 3.1623 | 通过 |
| 20 cm | heavy | pass | 0.9392 | 0.9515 | 0.9864 | 0.9686 | 3.6056 | 通过 |
| 40 cm | light | diagnostic_only | 0.7229 | 0.9996 | 0.7232 | 0.8392 | 69.0062 | 未通过 |
| 40 cm | moderate | diagnostic_only | 0.9730 | 0.9932 | 0.9795 | 0.9863 | 2.2361 | 通过 |
| 40 cm | heavy | pass | 0.9732 | 0.9835 | 0.9894 | 0.9864 | 2.8284 | 通过 |

按水深，20 cm 最稳定，3/3 通过；10 cm 与 40 cm 各 2/3 通过；5 cm 仅 1/3 通过。按雨强，moderate 的 IoU 中位数最高（0.9555）且边界最稳定；heavy 中 5 cm 和 10 cm 出现明显过分割，precision 分别仅 0.4830 和 0.7120；40 cm light 则出现欠分割，recall 为 0.7232。上述三项是当前主要失败模式。

Prompt prediction-side 状态与最终 GT 准确率并非一一对应：5 cm light、10 cm light、20 cm light 和 40 cm moderate 虽为 `diagnostic_only`，其中四项均达到离线分割条件；反之，5 cm heavy 和 10 cm heavy 的 Prompt 状态为 `pass`，SAM 2 mask 仍明显过分割。因此时序提示质量门控不能替代独立分割质量评价。

本矩阵与既有人工提示 held-out 样本不构成严格同帧配对，不能据此声称自动提示已达到或超过人工提示。后续如需定量比较，应冻结同一 RGB 后分别生成“人工提示”和“自动提示”两套候选，并保持 GT 最后独立读取。

关于 3 px：20 cm 的三项 IoU 均为 0.9392–0.9571、F1 均高于 0.9686，但 outer boundary P95 为 3.1623–4.0000 px。这说明 3 px 对像素岸线研究指标可能偏保守；但这里计算的是 SAM mask 对 Camera GT 外岸线误差，并非现有几何 gate 的 DEM 重投影自一致性指标，因此本阶段不据此修改 gate。应在 Phase 2D-C-8 使用更多配对证据重新设计门控。

生成结果保存在 `outputs/sam2_auto_prompt_gt_evaluation_seed301_frame149/`，由 `.gitignore` 排除，不进入提交。

## 结论边界

本阶段只评价 Camera 可见域的单帧 SAM 2 掩膜。它不能证明自动积水语义识别已经完成，也不能验证 DEM 水深、全局面积或体积。40 cm 场景中的 Camera 不可观测第二盆地仍不应由图像候选自动补全。

当前证据支持继续 Phase 2D-C-6 的失败样本诊断与固定规则改进，重点处理 heavy 下浅水过分割和 40 cm light 欠分割；在冻结统一规则并获得新的 held-out 证据之前，不进入正式 S5-S8。
