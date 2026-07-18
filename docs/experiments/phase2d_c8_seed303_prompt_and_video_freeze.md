# Phase 2D-C-8-3B：seed 303 自动提示与 SAM2 视频传播冻结

## 1. 实验目的与隔离

本阶段对 C8-3A 已冻结的 12 个 seed 303 anchor 自动生成提示，并在固定 frame 129–169 窗口运行 SAM 2 视频传播。

- 未人工打开 RGB 或调整提示；
- 自动提示沿用冻结的 C6C 固定规则；
- 每个样本仅运行一次 SAM 2；
- 未读取 Camera/DEM mask GT、真实水位、depth、area、volume 或 evaluation 输出；
- 未运行 ray–DEM、几何 prediction 或候选 quality gate；
- 输出仍为 `unknown_candidate`、`authoritative=false`、`eligible_for_downstream=false`。

## 2. 自动提示冻结

12 个 anchor 的 SHA-256 在第一份提示生成前一次性验证。自动提示结果：

| 样本 | 场景 | 雨强 | Prompt 状态 |
|---|---|---|---|
| c8c3_001 | 5 cm | light | diagnostic_only |
| c8c3_002 | 5 cm | moderate | pass |
| c8c3_003 | 5 cm | heavy | pass |
| c8c3_004 | 10 cm | light | diagnostic_only |
| c8c3_005 | 10 cm | moderate | diagnostic_only |
| c8c3_006 | 10 cm | heavy | pass |
| c8c3_007 | 20 cm | light | diagnostic_only |
| c8c3_008 | 20 cm | moderate | pass |
| c8c3_009 | 20 cm | heavy | pass |
| c8c3_010 | 40 cm | light | diagnostic_only |
| c8c3_011 | 40 cm | moderate | diagnostic_only |
| c8c3_012 | 40 cm | heavy | pass |

合计 6 pass、6 diagnostic_only、0 reject。依据既有研究协议，pass 和 diagnostic_only 均可进入离线 SAM 2 验证，reject 不允许启动传播。

首次串行提示调度因外部 SSH 工具 10 分钟上限中断，但 VMware 子进程继续完成第 3 份提示。恢复时保留并复核前三份完整输出，只对缺失的九份提示进行最多三路并行续跑；没有覆盖或重跑已完成提示。

## 3. SAM 2 视频传播

运行环境：

- SAM 2.1 Hiera Tiny；
- PyTorch 2.7.1+cu128；
- NVIDIA GeForce RTX 4060 Laptop GPU；
- 12 个样本 × 41 帧，共 492 帧；
- 所有 RGB window、anchor 和 prompt SHA-256 在首个 SAM 2 进程前通过。

传播结果：

| 样本 | 面积 CV | 相邻 mask IoU 最小值 | 相邻 mask IoU 中位数 | 耗时/s |
|---|---:|---:|---:|---:|
| c8c3_001 | 4.55% | 0.9427 | 0.9714 | 9.64 |
| c8c3_002 | 4.64% | 0.8961 | 0.9612 | 6.81 |
| c8c3_003 | 5.76% | 0.8794 | 0.9338 | 5.79 |
| c8c3_004 | 3.91% | 0.8731 | 0.9729 | 5.64 |
| c8c3_005 | 1.94% | 0.9383 | 0.9702 | 5.69 |
| c8c3_006 | 3.69% | 0.8733 | 0.9563 | 5.78 |
| c8c3_007 | 1.32% | 0.9620 | 0.9828 | 5.55 |
| c8c3_008 | 1.85% | 0.9567 | 0.9804 | 5.64 |
| c8c3_009 | 1.82% | 0.9473 | 0.9752 | 5.63 |
| c8c3_010 | 1.69% | 0.9793 | 0.9898 | 5.53 |
| c8c3_011 | 1.24% | 0.9537 | 0.9824 | 5.40 |
| c8c3_012 | 0.64% | 0.9721 | 0.9864 | 5.47 |

12 次 SAM 2 运行均成功，`sam2_rerun_count=0`，CUDA OOM 数量为 0。每次运行峰值 allocated 显存约 1,148.94 MiB。

## 4. 输出冻结

WSL 主输出：

`/home/wlkl/sam2_yujian_workspace/outputs/phase2d_c8_seed303_video_freeze/`

Windows 交换副本：

`D:/yujian_exchange/phase2d_c8_seed303_video_freeze/`

VMware ignored 输出：

`outputs/phase2d_c8_seed303_video_freeze/`

三端 `video_matrix_summary.json` 的 SHA-256 为：

`c3c2c4a643e24726c614e9f14042facc6040b83b8ac47bf239d4ee10b2c67590`

同步结果包含 492 个 NPY mask，没有 Ground Truth 文件；生成输出由 `.gitignore` 排除。

## 5. 当前结论与下一步

本阶段只证明 seed 303 的 GT-free 自动提示和 SAM 2 视频传播接口完整可运行。面积 CV 和相邻 mask IoU 是 prediction-side 时间一致性，不代表积水语义或水位准确。

下一阶段 C8-3C 将在 VMware 中先验证所有冻结 mask 哈希，再运行既有 ray–DEM 几何链路和已冻结的 C8 候选 gate。全部 prediction 冻结后，独立 evaluation 才允许读取 GT，最终判断候选 gate 是否通过 seed 303 确认。
