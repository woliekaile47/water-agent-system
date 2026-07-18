# Phase 2D-C-6C-1：自动提示统一修复规则冻结

## 目标

根据 Phase 2D-C-6B-4 的失败归因，冻结一套不依赖 Ground Truth、不得逐 case 调节的自动提示安全规则。此阶段只实现规则和合成单元测试，不在已审计的 12 个样本上重跑 temporal prediction 或 SAM 2。

## 固定规则

1. **高置信正点核心**：正点除满足组件内部距离和空间间隔外，还必须满足 `predicted_water_probability >= 0.50`。若不足 3 点，提示直接 `reject`，不以低置信雨滴动态补齐。
2. **Partial coverage box 扩展**：当 temporal gate 不是 `pass` 或 `reject` 时，box 的横纵 margin 分别取基础 12 px 与组件宽高 50% 的较大值，并限制最大 96 px。该 box 仍受既有面积和图像边缘安全约束。
3. **Partial gate 负点降级**：coverage 为 partial 时禁用 `dry_splash_track` 负点，只使用与图像外部背景连通的 known-nonwater ring，避免把弱观测水域误作负点。

规则通过独立配置 `configs/temporal_sam2_prompt_c6c.yaml` 启用，旧 `configs/temporal_sam2_prompt.yaml` 保持 Phase 2D-C-6B baseline 行为。核心实现对缺少新配置项的旧配置提供兼容默认值。

## 语义与限制

- 所有提示继续标记为 `unknown_candidate`、`authoritative=false`、`eligible_for_downstream=false`。
- `diagnostic_only` 不等于自动通过，`reject` 不得运行正式下游测量。
- 本阶段没有修改 temporal quality gate、3 px 几何 gate 或 SAM 2 Candidate 选择策略。
- 已审计 12 样本从此视作开发诊断集，不再作为 C6C 改进后的 held-out 结论来源。

## 下一阶段冻结验证

Phase 2D-C-6C-2 必须使用未参与 C6B 诊断的新 seed/frame，按 4 水深 × 3 雨强统一矩阵运行一次。GT 仍在 prompt 与 SAM 2 输出冻结后由独立 evaluation 读取，禁止逐 case 调参或失败后重跑。
