# Phase 2D-C-6C-2：固定自动提示规则的独立 seed 验证

## 实验协议

本阶段使用已冻结的 seed 302、frame 99 矩阵验证 Phase 2D-C-6C-1 固定规则。矩阵覆盖 4 个水深和 3 个雨强，共 12 个样本。样本在运行前未主观浏览 RGB，prompt 和 SAM 2 mask 均在读取 GT 前冻结并记录 SHA-256。

- Prompt 配置：`configs/temporal_sam2_prompt_c6c.yaml`
- SAM 2：SAM 2.1 Hiera Tiny，固定最高模型 score Candidate 1
- SAM 2 运行次数：12；重跑次数：0；CUDA OOM：0
- Prompt 状态：9 pass、3 diagnostic_only、0 reject
- GT 范围：独立 evaluation 只读取严格同状态 Camera water mask
- 未修改 temporal gate、几何 gate、Candidate 选择或 S5-S8

seed 302 曾用于更早的人工提示实验，但 C6C 固定规则来自 seed 301 自动提示失败审计；frame 99 未用于此前 frame 49/149 的人工标注。本结果可作为独立 seed 验证，但不冒充全新 seed 303 或真实视频外部验证。

## 评价结果

8/12 个样本同时满足既有离线研究条件：IoU ≥ 0.90、recall ≥ 0.90、outer boundary P95 ≤ 5 px。该条件只用于离线研究，不是 prediction-side gate。

| 水深 | 雨强 | Prompt | IoU | Precision | Recall | F1 | Outer P95 / px | 结果 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 5 cm | light | diagnostic_only | 0.9396 | 0.9753 | 0.9625 | 0.9688 | 2.0000 | 通过 |
| 5 cm | moderate | pass | 0.8741 | 0.8889 | 0.9813 | 0.9328 | 4.1231 | 未通过 |
| 5 cm | heavy | diagnostic_only | 0.5148 | 0.5149 | 0.9994 | 0.6797 | 29.0015 | 未通过 |
| 10 cm | light | pass | 0.8964 | 0.9948 | 0.9007 | 0.9454 | 7.0000 | 未通过 |
| 10 cm | moderate | pass | 0.9410 | 0.9594 | 0.9801 | 0.9696 | 2.8284 | 通过 |
| 10 cm | heavy | pass | 0.6433 | 0.6441 | 0.9981 | 0.7829 | 15.5451 | 未通过 |
| 20 cm | light | pass | 0.9428 | 0.9984 | 0.9442 | 0.9705 | 4.0000 | 通过 |
| 20 cm | moderate | pass | 0.9586 | 0.9861 | 0.9718 | 0.9789 | 2.8284 | 通过 |
| 20 cm | heavy | pass | 0.9235 | 0.9335 | 0.9885 | 0.9602 | 4.2656 | 通过 |
| 40 cm | light | diagnostic_only | 0.9657 | 0.9981 | 0.9675 | 0.9826 | 3.1623 | 通过 |
| 40 cm | moderate | pass | 0.9768 | 0.9948 | 0.9819 | 0.9883 | 2.2361 | 通过 |
| 40 cm | heavy | pass | 0.9637 | 0.9702 | 0.9931 | 0.9815 | 3.4282 | 通过 |

总体 Camera IoU 最小值 0.5148、中位数 0.9403、均值 0.8784、最大值 0.9768。Outer boundary P95 中位数为 3.7141 px。

## 固定规则效果与失败边界

40 cm light 的 box 范围问题在本批没有复现：IoU 为 0.9657，说明 partial coverage 下的组件尺度 box 扩展具有研究价值。但该样本 SAM 2 内部 score 仅 0.0233，仍获得较高 GT 指标，因此 SAM 2 模型 score 不能单独作为积水语义 quality gate。

浅水强雨问题仍然存在：5 cm heavy 和 10 cm heavy recall 接近 1，但 precision 分别只有 0.5149 和 0.6441，属于明显过分割。评价侧审计显示：

- 5 cm heavy 正点 GT 支持率为 0.80，box 完整覆盖 GT；
- 10 cm heavy 正点 GT 支持率为 0.60，box 完整覆盖 GT；
- 两项负点正确率均为 1.00；
- 错误主要位于 box 内。

这表明固定概率下限能避免部分低置信错误点，但不能区分高置信雨滴动态与真实水域；继续调整单帧阈值不足以安全解决问题。

5 cm moderate 仍为轻度过分割。10 cm light 的主要问题是局部边界和欠覆盖：IoU 0.8964、outer P95 7 px。20 cm 和 40 cm 在三种雨强下均达到离线条件。

与 seed 301 的 C6B 结果相比，C6C 的 IoU 均值由 0.8666 变为 0.8784，outer P95 均值由 11.8687 px 变为 6.7016 px，但两批 seed/frame 不同，不能将该差异解释为严格配对提升。两批均为 8/12 通过，浅水 heavy 过分割是稳定复现的主要风险。

## 冻结结论

Phase 2D-C-6 已形成可重复、全自动、GT-free 的单帧提示与 SAM 2 mask 基线，但尚未达到所有天气条件下的可靠积水语义识别。所有结果仍为 `unknown_candidate`、`authoritative=false`、`eligible_for_downstream=false`。

下一阶段进入 Phase 2D-C-7：在不修改本批结果的前提下研究连续帧 SAM 2 传播、mask 共识、岸线和水位时间稳定性。其核心目标是使用跨帧一致性抑制浅水强雨下的瞬时过分割，而不是继续在本批 held-out 样本上调单帧阈值。
