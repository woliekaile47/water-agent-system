# Phase 2D-C-6C-2A：统一提示规则验证矩阵冻结

## 冻结规则

为验证 Phase 2D-C-6C-1 的统一 GT-free 提示规则，冻结 4 个水深 × 3 个雨强的 12 样本矩阵。所有样本使用 seed 302 和固定 frame 99，未打开 RGB 进行主观筛选，未读取任何 Camera mask、水位、DEM、depth、面积、体积或 evaluation 输出。

选择顺序为：优先寻找覆盖全部 4 水深和 3 雨强、且未用于 C6B 自动提示诊断的共同 seed；现有数据中 seed 302 满足条件。固定 frame 99 是序列中点附近的确定性索引，并避开此前人工提示使用的 frame 49/149。

## 数据独立性边界

seed 302 曾用于较早的人工提示多帧实验，因此不能表述为“从未被任何历史实验接触的全新序列”。但 C6C 规则来自 seed 301 的自动提示失败审计，本阶段没有查看 seed 302/frame 99 RGB 或 GT 后选择样本，故可作为 C6C 规则的独立 seed 验证。更严格的最终外部验证仍应增加全新 seed 303 或真实视频。

## 固定协议

- 矩阵配置：`configs/phase2d_c6c2_heldout_matrix.yaml`
- Prompt 配置：`configs/temporal_sam2_prompt_c6c.yaml`
- SAM 2：每个允许运行的 prompt 只运行一次，固定选择最高模型分数 Candidate 1。
- Prompt `reject` 不运行 SAM 2；`diagnostic_only` 只允许研究诊断，不进入下游。
- 在 prompt、SAM 2 mask 与哈希全部冻结前禁止读取 GT。
- 不允许逐 case 调参、失败后重跑或回退到人工提示。

## 下一步

Phase 2D-C-6C-2B 先在 VMware 运行 12 组 temporal prediction 和 prompt 生成，汇总 pass/diagnostic/reject。随后仅对允许的候选通过 WSL GPU 运行一次 SAM 2，并最后由独立 evaluation 读取 Camera GT。
