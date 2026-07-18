# Phase 2D-C-6B-2：自动提示 SAM 2 held-out 一次性运行

## 1. 实验目的

本阶段验证 Phase 2D-C-6B 的 GT-free 自动提示生成器能否在固定、多水深、多雨强样本上稳定生成 SAM 2 Box、正点和负点，并将非 reject 提示交给现有 SAM 2 runner 各运行一次。

本阶段不读取 Ground Truth，不计算 IoU、水位、面积或体积误差，也不根据 SAM 2 输出重跑时序 prediction、修改提示或调整阈值。所有输出仍是 `unknown_candidate`，不是自动积水语义识别结果。

## 2. Held-out 矩阵冻结

冻结规则：

- 4 个场景：5、10、20、40 cm；
- 3 个雨强：light、moderate、heavy；
- 固定新组合：seed 301、frame 149；
- 共 12 个样本；
- 不打开 RGB 主观挑选；
- 不读取任何 Ground Truth；
- 每张 RGB 在 prediction 前以 SHA-256 校验。

此前 Phase 2D-C-5 使用的是 `seed301/frame49`、`seed302/frame49` 和 `seed302/frame149`。本次 `seed301/frame149` 未进入 C5 的 36 样本，且在冻结矩阵前未用于 C6B 自动提示规则调节。

矩阵保存在 `configs/phase2d_c6b2_heldout_matrix.yaml`。case 名称只用于数据定位和实验汇总；prediction 函数不接收或解析 nominal depth。

## 3. Prediction-only 数据流

```text
continuous RGB frames
  -> existing temporal prediction (full + shuffled)
  -> existing temporal quality gate
  -> water probability / water mask / unknown mask / dry-splash tracks
  -> fixed automatic prompt generator
  -> prompt quality gate
  -> frozen automatic_prompt.json
  -> existing WSL SAM 2 runner, once per non-reject sample
  -> frozen raw unknown-candidate mask
```

新增 frames-to-prompt 入口不导入 `src.evaluation`，不接受 GT mask、water level、depth、area 或 volume 参数。

## 4. 自动提示结果

12 个样本全部完成提示生成：

- `pass`：7；
- `diagnostic_only`：5；
- `reject`：0；
- 每个样本均生成 5 个正点、8 个负点；
- 每个样本负点均覆盖 8/8 方向；
- `ground_truth_used=false`；
- `eligible_for_downstream=false`。

5 个 `diagnostic_only` 样本为 5/10/20/40 cm light 以及 40 cm moderate，原因均为 `temporal_quality_gate_not_pass`。该状态允许本阶段继续执行离线 SAM 2 诊断，但不能进入正式几何或预警链路。

## 5. SAM 2 一次性运行

固定环境：

- SAM 2.1 Hiera Tiny；
- config：`configs/sam2.1/sam2.1_hiera_t.yaml`；
- checkpoint：`sam2.1_hiera_tiny.pt`；
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU；
- candidate policy：固定选择最高模型 score；
- 每个非 reject 提示只运行一次；
- `sam2_run_count=12`；
- `rerun_count=0`；
- CUDA OOM：0；
- 12/12 均确认选择最高 score candidate。

模型 score 表示 SAM 2 对候选 mask 质量的内部预测，不是 water probability。

## 6. 非 GT 输出汇总

| sample | 场景 | 雨强 | prompt gate | SAM2 score | mask pixels | components | holes |
|---|---|---|---|---:|---:|---:|---:|
| c6b2_001 | 5 cm | light | diagnostic_only | 0.878906 | 1,738 | 2 | 0 |
| c6b2_002 | 5 cm | moderate | pass | 0.820312 | 1,952 | 1 | 0 |
| c6b2_003 | 5 cm | heavy | pass | 0.730469 | 3,638 | 8 | 8 |
| c6b2_004 | 10 cm | light | diagnostic_only | 0.855469 | 3,435 | 1 | 0 |
| c6b2_005 | 10 cm | moderate | pass | 0.859375 | 3,651 | 1 | 0 |
| c6b2_006 | 10 cm | heavy | pass | 0.738281 | 5,069 | 11 | 6 |
| c6b2_007 | 20 cm | light | diagnostic_only | 0.882812 | 7,536 | 1 | 0 |
| c6b2_008 | 20 cm | moderate | pass | 0.875000 | 7,703 | 1 | 0 |
| c6b2_009 | 20 cm | heavy | pass | 0.812500 | 8,227 | 3 | 0 |
| c6b2_010 | 40 cm | light | diagnostic_only | 0.613281 | 15,668 | 4 | 18 |
| c6b2_011 | 40 cm | moderate | diagnostic_only | 0.886719 | 21,358 | 2 | 2 |
| c6b2_012 | 40 cm | heavy | pass | 0.824219 | 21,786 | 1 | 2 |

总体：

- mask 面积最小值：1,738 px；
- mask 面积中位数：6,302.5 px；
- mask 面积最大值：21,786 px；
- score 最小值：0.613281；
- score 中位数：0.839844；
- score 最大值：0.886719；
- 最大 GPU allocated memory：556.003 MiB；
- 无 CUDA OOM。

面积随场景水深总体上升只是非 GT 的输出分布观察，不能据此证明 mask 正确，也不能用于选择或修改提示规则。

## 7. 冻结与可追溯性

每个样本保存：

- frozen automatic prompt；
- prompt gate 状态；
- reference RGB SHA-256；
- SAM 2 raw mask；
- raw mask SHA-256；
- SAM 2 summary SHA-256；
- candidate score、面积、组件数和孔洞数；
- GPU 和耗时信息。

VMware 自动提示输出：

```text
outputs/temporal_sam2_prompt_heldout_seed301_frame149/
```

冻结 SAM 2 输出：

```text
outputs/sam2_auto_prompt_heldout_seed301_frame149/
```

两类生成输出均由 `.gitignore` 排除，不进入 Git。

## 8. Runner 元数据说明

现有 WSL runner 的静态 `result_note` 仍包含 “manually guided” 文案，这是早期人工提示实验遗留的说明字符串。本批真实输入由 `prompt_source=temporal_water_evidence_v1` 标识，提示坐标为自动生成。

该旧文案不改变冻结 mask，但不能作为本批提示来源的权威元数据。后续可在不重跑本批输出的前提下，让 runner 根据 `prompt_source` 生成更准确的说明。

## 9. 当前结论与限制

当前证据支持：

1. 自动提示链路可以在 12/12 固定样本上产生合法提示；
2. 现有 SAM 2 runner 能读取自动提示 schema 并稳定运行；
3. 12/12 固定采用最高 score candidate；
4. dry 安全性已在 C6B smoke test 中验证；
5. 传输和输出具备 SHA-256 可追溯性。

当前证据不支持：

- 宣称自动识别积水成功；
- 宣称 mask 与真实水域一致；
- 修改 geometry 或 boundary gate；
- 将 diagnostic/rejected candidate 作为 authoritative measurement；
- 接入 S5-S8、Agent、数据库或 Dashboard。

下一小阶段应对本批已冻结的 12 个 raw mask 进行独立 Ground Truth evaluation。GT 只能由 evaluation 模块读取，且不得反向修改时序模块、自动提示、SAM 2 candidate 或本批冻结输出。
