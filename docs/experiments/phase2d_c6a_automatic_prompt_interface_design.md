# Phase 2D-C-6A：时序视觉到 SAM 2 自动提示接口设计

## 1. 阶段目标

本阶段只冻结接口与算法规则，不运行新的 SAM 2 大矩阵，也不修改现有 prediction、Ground Truth evaluation 或正式 quality gate。目标是把现有 GT-free 时序视觉证据转换为可由 WSL SAM 2 runner 直接读取的 Box、正点和负点，从而在 Phase 2D-C-6B 替代人工点击。

自动提示仍只产生 `unknown_candidate`，不能直接称为积水语义识别结果。只有后续独立评价和质量门控通过后，候选结果才具备继续进行几何诊断的资格。

## 2. 已有能力盘点

VMware 仓库已有以下 prediction-side 能力：

- `src/perception/temporal_water_pipeline.py`：只读取连续 RGB 帧，输出时序候选、track、classification 和稀疏水域证据；
- `src/perception/temporal_water_evidence.py`：输出 `predicted_water_probability`、`predicted_water_mask`、`predicted_unknown_mask` 和 `evidence_count_map`；
- `src/perception/temporal_water_quality_gate.py`：检查输入帧数、动态 track、时序顺序敏感度、mask 稳定性、unknown 比例和连通域；
- `src/integration/synthetic_visual_to_depth.py`：保持 prediction 与 Ground Truth evaluation 分离；
- `src/fusion/sam2_shoreline_geometry_adapter.py`：读取冻结 SAM 2 mask、岸线和人工正点，进入既有几何诊断；
- `scripts/run_sam2_shoreline_geometry_diagnostic.py`：执行 GT-free ray–DEM、水位估计、低地重建和 prediction-side gate。

WSL 工作区已有 `run_prompted_mask_test.py`，支持扩展提示 JSON：

- `image_width`、`image_height`；
- `prompt_source`；
- `semantic_label = unknown_candidate`；
- `authoritative = false`；
- `box_xyxy`；
- `positive_points_xy`；
- `negative_points_xy`。

因此 Phase 2D-C-6B 不需要修改 SAM 2 官方仓库，也不需要改变模型权重或 checkpoint。

## 3. 目标数据流

```text
continuous RGB frames
  -> temporal prediction
  -> water probability / water mask / unknown mask / tracks
  -> automatic prompt generator
  -> frozen prompt JSON
  -> SAM 2 highest-score candidate (fixed policy)
  -> frozen raw mask
  -> existing shoreline and DEM geometry chain
  -> independent Ground Truth evaluation
```

时序 mask 是稀疏动态证据，不应被直接冒充完整水域。它的职责是为 SAM 2 提供提示，而 SAM 2 的职责是扩展到当前 RGB 帧中外观一致的候选区域。

## 4. 自动提示生成规则

所有样本使用统一配置，禁止按水深、雨强、seed 或评价结果逐 case 调参。

### 4.1 参考帧

- 离线对照实验使用输入 manifest 预先冻结的 frame index；
- 在线滑动窗口使用窗口最后一帧；
- 禁止通过 Camera IoU、Ground Truth 或人工浏览选择“最好”的帧；
- 输出必须记录 RGB SHA-256 和 frame index。

### 4.2 主候选连通域

1. 从 `predicted_water_mask` 提取 8 邻域连通域；
2. 按组件内 water probability 总和排序，面积作为第二排序键；
3. 排序必须确定性一致，最终以最小 label 作为平局规则；
4. unknown 像素不得加入候选域；
5. 多个接近的候选组件不得静默合并，应记录歧义并由 prompt gate 判定。

### 4.3 Box

1. 取主候选域的最小外接矩形；
2. 使用统一的固定像素 margin 扩展；
3. 裁剪至图像边界；
4. Box 必须包含全部正点和主候选域；
5. Box 大量接触图像边缘或面积比例异常时标记 `diagnostic_only` 或 `reject`。

### 4.4 正点

1. 正点只能位于主候选域内部；
2. 正点优先选择 distance transform 的内部极值，并以 water probability 作为第二评分；
3. 使用确定性的最远点采样，避免多个点集中在同一涟漪中心；
4. 正点必须与候选域边界保持统一的最小距离；
5. 正点数量固定在统一配置范围内，默认目标 5 个、最低 3 个；
6. 若安全内部像素不足，不放宽到 unknown 或边界像素，直接拒绝生成提示。

### 4.5 负点

负点优先级如下：

1. 高置信 `dry_splash` track 中心；
2. 主候选域外侧、`~water & ~unknown` 的已知非水环带；
3. Box 四周不同方位的确定性分层采样。

负点不得落入 water mask、unknown mask、主候选域或图像外。默认目标 8 个，并要求覆盖候选域多个方向。不能获得足够可信负点时，prompt gate 必须拒绝，而不是把未知路面当成负样本。

## 5. 自动提示 JSON

建议输出：

```json
{
  "schema_version": "phase2d_c6_prompt_v1",
  "image_path": "...",
  "image_sha256": "...",
  "image_width": 640,
  "image_height": 360,
  "frame_index": 149,
  "prompt_source": "temporal_water_evidence_v1",
  "semantic_label": "unknown_candidate",
  "authoritative": false,
  "box_xyxy": [x1, y1, x2, y2],
  "positive_points_xy": [[x, y]],
  "negative_points_xy": [[x, y]],
  "ground_truth_used": false,
  "eligible_for_downstream": false
}
```

另存 `automatic_prompt_diagnostics.json`，记录组件数量、候选面积、概率质量、unknown 接触、Box 边界接触、点到边界距离、点来源、拒绝原因和全部输入 SHA-256。

## 6. Prompt quality gate

至少包含以下拒绝条件：

- temporal quality gate 为 `reject`；
- predicted water mask 为空；
- 主候选域过小或非有限；
- 候选组件过多且无法唯一选择；
- 正点少于最低数量；
- 正点接触候选边界或落入 unknown；
- 可信负点不足；
- 负点落入 water 或 unknown；
- Box 无效、面积异常或大面积触碰图像边缘；
- 输入尺寸、frame index 或 SHA-256 不一致；
- 任意数组含 NaN/Inf。

输出状态使用 `pass / diagnostic_only / reject`。无论状态如何，本阶段 `eligible_for_downstream` 均保持 `false`。

## 7. Ground Truth 隔离

自动提示模块的函数签名只允许接收 prediction artifacts 和 RGB，不接受 case depth、water level、Camera GT mask、DEM GT mask、depth GT、area 或 volume。

需要测试保证：

- 改变或删除 Ground Truth 不影响自动提示结果；
- 不导入 evaluation 模块；
- 不根据 SAM 2 与 GT 的 IoU 选择点、Box 或 candidate；
- SAM 2 candidate 使用固定最高模型分数策略；
- prediction 冻结后才允许独立 evaluation 读取 GT。

## 8. VMware 与 WSL 边界

- VMware 负责运行时序预测和自动提示生成；
- 提示 JSON、RGB 和 custody manifest 通过 Windows 交换目录传入 WSL；
- WSL 只运行现有 SAM 2 runner；
- checkpoint、SAM 2 官方源码和虚拟环境不进入 water-agent-system Git；
- WSL 输出 raw mask 后记录 SHA-256，再复制回 VMware 进入既有岸线几何链路；
- 传输前后必须逐文件校验 SHA-256。

## 9. Phase 2D-C-6B 建议新增文件

```text
configs/temporal_sam2_prompt.yaml
src/vision/generate_temporal_sam2_prompt.py
scripts/generate_temporal_sam2_prompt.py
tests/test_temporal_sam2_prompt.py
tests/test_temporal_sam2_prompt_no_gt_leakage.py
docs/experiments/phase2d_c6b_automatic_prompt_baseline.md
```

现有时序、SAM 2 岸线、几何和独立评价模块继续复用，不在 C6B 重写。

## 10. Phase 2D-C-6B 验收条件

1. dry 输入不会生成水域正点；
2. 相同输入与配置输出完全确定；
3. 所有点和 Box 均位于 640×360 图像范围；
4. 正点只位于 water candidate 安全内部；
5. 负点不落入 water 或 unknown；
6. 多组件、空 mask、边缘截断和低证据输入安全拒绝；
7. Ground Truth 隔离测试通过；
8. 在未参与规则设计的新 held-out 矩阵上一次性运行自动提示和 SAM 2；
9. 自动提示结果与人工提示结果分别报告，不使用 GT 挑选更好方案；
10. 通过 gate 前不接入正式 S5-S8。

## 11. 当前结论

现有系统已经具备自动提示所需的全部 prediction-side上游证据和 SAM 2 下游接口。下一步应实现一个小型、确定性、GT-free 的提示生成器，而不是重新训练 SAM 2 或修改几何算法。最大的技术风险是时序证据稀疏、unknown 区域过大和可信负点不足；这些情况必须由 prompt gate 拒绝，不能通过猜测补全。
