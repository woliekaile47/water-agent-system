# Phase 2D-C-6B：时序视觉自动生成 SAM 2 提示基线

## 1. 阶段目标

本阶段实现 Phase 2D-C-6A 已冻结的 prediction-side 接口：把已有时序视觉概率图、水域候选 mask、unknown mask、事件分类和 temporal quality gate 转换为可供现有 SAM 2 runner 读取的 Box、正点和负点。

本阶段只完成生成器、质量门控、单元测试和轻量 smoke test；没有运行新的 SAM 2 held-out 矩阵，也没有读取 Ground Truth 选择点位或参数。输出语义始终为 `unknown_candidate`，`authoritative=false`，`eligible_for_downstream=false`。

## 2. 输入与 Ground Truth 隔离

生成器只读取：

- `predicted_water_probability.npy`；
- `predicted_camera_water_mask.png` 或兼容的 `predicted_water_mask.png`；
- `predicted_camera_unknown_mask.png` 或兼容的 `predicted_unknown_mask.png`；
- `event_classifications.json` 或 `temporal_diagnostics.json` 中的 prediction-side classification；
- `visual_quality_gate.json` 或兼容的 `quality_gate.json`；
- 预先冻结的参考 RGB、frame index 和统一配置。

函数签名不接受 Camera/DEM Ground Truth mask、真实水位、depth、area、volume 或 nominal depth。源码不导入 `src.evaluation`。独立测试通过修改无关的损坏 GT 文件，确认输出不发生变化。

## 3. 固定提示生成规则

1. 使用 8 邻域提取 temporal water mask 连通域；
2. 按 probability mass、面积、label 的固定顺序选择主组件；
3. 若次级组件的 probability mass 达到主组件固定比例，标记歧义并 reject；
4. Box 为主组件外接矩形加固定 12 px margin，并裁剪到图像范围；
5. 正点仅从主组件安全内部选择，优先 distance transform 深处，再按 probability 和确定性最远点采样分散；
6. 负点仅来自主组件外部、与图像边缘相连的已知非水区域；
7. 内部孔洞和 unknown 像素不能作为负点；
8. 合格 `dry_splash` track 中心优先，但必须处于固定岸线环带；随后按八个方向用已知非水环带补齐；
9. temporal gate reject、空 mask、组件歧义、点数不足、方向覆盖不足、Box 异常或非有限输入均安全 reject。

所有阈值保存在 `configs/temporal_sam2_prompt.yaml`，不按水深、雨强、seed 或 case 调整。

## 4. 输出与可追溯性

CLI 输出：

- `automatic_prompt.json`；
- `automatic_prompt_diagnostics.json`；
- `automatic_prompt_preview.png`。

提示 JSON 使用 `phase2d_c6_prompt_v1`，记录 RGB SHA-256、frame index、Box、正负点、prompt gate、全部已读取 prediction artifact 的 SHA-256，以及 `ground_truth_used=false`。

## 5. 20 cm water smoke test

只读取既有 prediction artifacts：

```text
outputs/synthetic_visual_to_depth_integration/
  sim_water_20cm_001/heavy/seed_43/
```

参考帧为 `frame_000059.png`，RGB SHA-256 为：

```text
1ff9e6601a044e8764276247f0bc05a78918a168984a4c6433acf1cb4419b4b0
```

结果：

| 指标 | 结果 |
|---|---:|
| temporal 主组件数 | 1 |
| 主组件面积 | 9,023 px |
| Box | `[264, 119, 447, 220]` |
| 正点数 | 5 |
| 负点数 | 8 |
| 负点方向覆盖 | 8/8 sectors |
| prompt quality status | `pass` |
| ground_truth_used | `false` |
| eligible_for_downstream | `false` |

该结果只说明接口能够从现有 GT-free 时序证据确定性地产生结构合法的 SAM 2 提示，不说明 SAM 2 已正确分割积水，也不说明水深链路已经通过。

## 6. Dry safety smoke test

只读取既有 `sim_dry_baseline_001/heavy/seed_43` prediction artifacts。结果为：

- `prompt_quality_status=reject`；
- 正点数为 0；
- 负点数为 0；
- Box 为 `null`；
- 主要原因包含 `predicted_water_mask_empty`；
- `ground_truth_used=false`。

因此 dry 空水域候选不会被转换成伪水域正点。

## 7. 测试覆盖

新增测试覆盖：

- 相同输入的确定性输出；
- 正点只在主候选安全内部；
- 负点不落入 water 或 unknown；
- dry 空 mask 安全 reject；
- probability mass 主组件选择；
- 近似等权组件歧义 reject；
- temporal gate reject 继承；
- 输入数组不被原地修改；
- `dry_splash` 负点方向去重与环带补齐；
- 内部孔洞不作为负点；
- prediction 接口及 CLI 无 Ground Truth/evaluation 依赖；
- 无关 GT 文件变化不影响结果。

## 8. 当前限制与下一步

本阶段没有：

- 运行 SAM 2 推理；
- 使用 GT 评价自动提示效果；
- 修改 SAM 2 官方仓库、模型或 checkpoint；
- 修改 Phase 2B/C3C 几何算法或正式 quality gate；
- 修改 3 px boundary 阈值；
- 接入 S5-S8、Agent、数据库或 Dashboard。

下一小阶段应冻结一组未参与规则设计的样本，先批量生成自动提示并审查 prompt gate 分布，然后在不读取 GT 的情况下对 gate 允许的样本各运行一次 SAM 2。SAM 2 raw mask 冻结后才能进入独立 GT evaluation；不得根据评价结果逐 case 修改提示规则。
