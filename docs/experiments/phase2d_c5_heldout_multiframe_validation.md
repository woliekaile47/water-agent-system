# Phase 2D-C-5：SAM 2 held-out 多样本与多帧验证

## 1. 实验目的与边界

本阶段验证人工提示 SAM 2 可见水域候选经过既有 Camera 岸线、ray–DEM、水位估计和 DEM 重建链路后，能否在多水深、多雨强、多 seed 和第二固定时间点上保持稳定。

本阶段仍属于离线研究验证：

- SAM 2 的 Box、正点和负点由人工提供，不代表已经实现自动积水语义识别；
- prediction 只读取 RGB、冻结的 SAM 2 mask、人工提示、相机参数和 dry Ground DEM；
- Ground Truth 只由独立 evaluation 读取；
- rejected 或 partially accurate 结果不作为 authoritative measurement；
- 结果未接入正式 S5-S8、Agent、数据库或 Dashboard；
- 未修改正式 quality gate 阈值。

## 2. 实验矩阵与确定性选样

累计评价 36 个样本：

| 子矩阵 | 水深 | 雨强 | 固定帧 | 样本数 |
|---|---|---|---:|---:|
| seed 301 | 5/10/20/40 cm | light/moderate/heavy | 49 | 12 |
| seed 302 | 5/10/20/40 cm | light/moderate/heavy | 49 | 12 |
| seed 302 多帧确认 | 5/10/20/40 cm | light/moderate/heavy | 149 | 12 |

frame 49 和 frame 149 均在打开 RGB 和 Ground Truth 之前按固定规则确定。frame 149 是 0–199 帧序列的确定性后段时间点，不通过浏览图像选择“最好”的帧。

## 3. 严格冻结与 Ground Truth 隔离

每个样本依次执行：

1. 冻结 RGB 路径与 SHA-256；
2. 用户只查看 RGB，人工保存一次提示；
3. SAM 2 prompted segmentation 只运行一次；
4. 冻结 prompt JSON 和 `prompted_mask_raw.npy`；
5. 从 raw mask 选择正点支持最多的主连通域；
6. 使用 `RETR_EXTERNAL + CHAIN_APPROX_NONE` 提取外岸线，并按弧长采样 128 点；
7. prediction-side 几何链路运行一次并记录结果 SHA-256；
8. prediction 冻结后，独立 evaluation 才读取严格同场景 Ground Truth；
9. evaluation 再次确认 prediction artifacts 未变化。

没有使用 Ground Truth 修改提示、选择 SAM 2 candidate、修正岸线、选择 basin、估计水位或调节 gate。

## 4. 36 样本总体结果

水位目标采用物理误差 3 cm；它与图像边界 P95 的像素阈值不是同一量纲。

- 36/36 样本水位绝对误差不超过 3 cm；
- 最大水位绝对误差：0.88198 cm；
- 中位水位绝对误差：0.36674 cm；
- 平均水位绝对误差：0.36858 cm；
- 独立评价状态：22 个 pass、1 个 fail、13 个 partially accurate；
- 冻结的原始 quality gate：19 个 pass、17 个 reject。

按水深汇总：

| 水深 | 样本数 | Camera IoU 中位数 | 水位最大误差 | 面积相对误差中位数 | 体积相对误差中位数 |
|---:|---:|---:|---:|---:|---:|
| 5 cm | 9 | 0.918691 | 0.88198 cm | 2.53% | 6.34% |
| 10 cm | 9 | 0.938854 | 0.61662 cm | 3.38% | 7.67% |
| 20 cm | 9 | 0.954710 | 0.58555 cm | 2.12% | 3.84% |
| 40 cm | 9 | 0.972310 | 0.58747 cm | 18.05% | 3.00% |

40 cm 面积误差仍主要来自 Camera 不可观测的第二独立盆地。可见主水域可以准确估计，但面积和体积语义必须区分 Camera 可见范围结果与全局场景结果。

## 5. seed 302 / frame 149 多帧确认结果

| 水深/cm | 雨强 | Camera IoU | 水位误差/cm | 面积误差 | 体积误差 | Boundary P95/px | 冻结 gate | 独立评价 |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 5 | light | 0.934843 | 0.1600 | 2.53% | 6.34% | 2.000 | pass | pass |
| 5 | moderate | 0.827654 | 0.0876 | 0.63% | 3.52% | 7.000 | reject | partially accurate |
| 5 | heavy | 0.864670 | 0.3555 | 6.33% | 13.81% | 3.000 | reject | partially accurate |
| 10 | light | 0.959016 | 0.3669 | 3.38% | 7.33% | 2.000 | pass | pass |
| 10 | moderate | 0.928647 | 0.3842 | 3.38% | 7.67% | 3.000 | pass | pass |
| 10 | heavy | 0.938854 | 0.5028 | 4.92% | 9.96% | 3.000 | pass | pass |
| 20 | light | 0.958951 | 0.5526 | 4.10% | 5.72% | 2.828 | pass | pass |
| 20 | moderate | 0.937782 | 0.4487 | 2.97% | 4.66% | 4.243 | reject | pass |
| 20 | heavy | 0.946737 | 0.5856 | 4.10% | 6.06% | 3.000 | pass | pass |
| 40 | light | 0.972294 | 0.5870 | 18.72% | 3.96% | 3.000 | reject | partially accurate |
| 40 | moderate | 0.970435 | 0.2691 | 17.59% | 2.10% | 3.941 | reject | partially accurate |
| 40 | heavy | 0.972685 | 0.4522 | 18.18% | 3.17% | 3.162 | reject | partially accurate |

5 cm moderate/heavy 再次说明浅水视觉岸线仍是主要风险。即使稳健水位估计误差较小，Camera mask 范围和局部岸线形状仍可能不足以支持 authoritative measurement。

## 6. 3/4/5 px quality gate 对照

对照实验只替换 boundary P95 的候选阈值，并保留所有非边界拒绝原因，包括 Camera 重投影一致性、盆地歧义、Camera 不可观测范围、NaN/Inf 和物理最大水深限制。

| Boundary P95 候选阈值 | Pass | Reject | 独立评价误拒 | 不安全放行 |
|---:|---:|---:|---:|---:|
| 3 px | 19 | 17 | 5 | 0 |
| 4 px | 23 | 13 | 1 | 0 |
| 5 px | 24 | 12 | 0 | 0 |

在当前 36 样本矩阵中，5 px 是新的研究候选；但该阈值由本批证据提出，不能直接在同一批数据上完成正式确认。正式配置继续保持 3 px，直至获得未参与候选选择的独立确认矩阵或完成正式门控设计评审。

## 7. 20 cm moderate 边界尾部审计

seed 302 / frame 149 / 20 cm moderate 在 4 px 候选下仍被误拒，因此进行了不重跑 prediction 的离线审计：

- observed mask 与 reprojected mask 的外岸线 P50：1.0 px；
- 外岸线 P95：5.0 px；
- 外岸线距离超过 4 px 的比例：7.38%；
- 超过 4 px 的像素集中区域约占整幅图像 3.81%；
- observed mask 与严格同帧 GT 的外岸线 P95：4.47 px；
- Camera mask IoU：0.937782；
- 水位绝对误差：0.4487 cm；
- 面积相对误差：2.97%；
- 体积相对误差：4.66%。

这表明误差是局部岸线形状尾部，而不是统一相机平移或整体 ray–DEM 投影错误。稳健水位估计对这类局部尾部具有容忍度，但 boundary gate 仍应保留为视觉/几何质量诊断，不能仅因水位误差小而删除。

## 8. 可观测性与结果语义

后续正式接口必须至少区分：

- `camera_visible_estimate`：只对 Camera 有证据的可见水域负责；
- `global_scene_estimate`：只有所有候选盆地均可观测或被其他传感器约束时才允许标记完整；
- `observable_lower_bound`：存在不可观测候选盆地时，面积和体积只能作为可见范围下界；
- `eligible_for_downstream`：人工提示或 gate reject 的结果保持 `false`。

不得用 DEM 低洼先验自动补全没有 Camera seed 的独立盆地。

## 9. 复现与输出管理

主要冻结输出包括 seed 301/frame 49、seed 302/frame 49 和 seed 302/frame 149 的 prediction 与独立评价目录。frame 149 prediction 冻结清单 SHA-256 为 `f0916cbe95243d66a6bc834fb20c301d5c8fc2b88469d2702c5b55beb78364f8`。

36 样本校准 JSON SHA-256 为 `18e708610094f9beb267c7656f9dbe464ee563b7c9e9e8673399b893a3bede89`。

自动生成的 NPY、PNG、JSON、CSV 和日志由 `.gitignore` 排除，不进入 Git。Windows 交换副本位于 `D:/yujian_exchange/sam2_multiframe_seed302_frame149/`。

## 10. 当前结论与下一阶段

当前证据支持以下结论：

1. 相机内外参、ray–DEM 求交、稳健水位估计和 DEM 低地重建在人工提示候选下基本正确；
2. 3 cm 水位目标在本批 36 个仿真样本上全部满足；
3. 5 cm 浅水视觉边界和 40 cm 全局不可观测盆地仍是主要风险；
4. 3 px boundary gate 对本批数据偏保守，5 px 可作为后续独立确认候选，但本阶段不修改正式阈值；
5. 人工提示 SAM 2 不能作为自动积水识别完成的证据。

下一阶段为 Phase 2D-C-6：由时序视觉模块自动生成 SAM 2 Box、正点和负点，并在不读取 Ground Truth 的条件下复用相同的冻结、prediction 和独立评价流程。
