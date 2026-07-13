# Phase 2D-C-3D：SAM 2 岸线几何结果独立 Ground Truth 评价

## 1. 实验边界

本阶段只评价 Phase 2D-C-3C 已保存的 prediction，没有重新运行 SAM 2，
没有重新运行或修改几何 prediction，也没有改变提示、候选、岸线、seed、
basin、水位、quality gate 或阈值。Ground Truth 仅由
`src/evaluation/evaluate_sam2_shoreline_geometry_gt.py` 读取，评价结果不具备
authoritative measurement 或正式下游预警语义。

评价对象固定为：

- case：`sim_water_5cm_001`
- sequence：`sim_water_5cm_001/heavy/seed_43`
- frame：`frame_000059.png`（640×360）
- prediction：`outputs/sam2_shoreline_geometry/frame_000059/`

## 2. Ground Truth 来源与一致性

Camera mask 来源为
`data/simulation/sim_water_5cm_001/ground_truth/camera_water_mask_gt.png`；动态序列
`data/simulation_dynamic/sim_water_5cm_001/heavy/seed_43/ground_truth/water_mask.png`
通过 sequence manifest 指回同一静态水域状态，二者逐像素相同。

DEM mask、depth、Ground DEM、water level、area 和 volume 分别来自该 case 的
`ground_truth/` 与 `manifest.json`。检查结果：

- Camera GT 为 640×360 二值图，水域 1,761 pixels；
- DEM mask、depth map 和 Ground DEM 均为 70×180；
- DEM 水域 158 cells，depth 有限、非负，mask 外严格为 0；
- 分辨率积分得到 area 1.58 m²、volume 0.0395474357 m³，与 manifest 和 metadata 一致；
- GT water level 为 -0.3938947141 m。

## 3. Camera mask 与外岸线评价

| 指标 | 结果 |
|---|---:|
| predicted / GT pixels | 2,795 / 1,761 |
| intersection / union | 1,733 / 2,823 |
| IoU | 0.613886 |
| precision | 0.620036 |
| recall | 0.984100 |
| F1 | 0.760755 |
| false positive / false negative | 1,062 / 28 pixels |
| 面积像素相对误差 | 58.72% |
| 连通域数量 | 1 |

预测几乎覆盖全部 GT 水域，但额外覆盖大量普通路面或涟漪/反射区域。因此问题是
明显的范围过扩，而不是水域主体漏检。

外岸线评价排除内部孔洞，仅使用与外部背景相邻的主体外轮廓：

| 边界方向 | P50 | P95 |
|---|---:|---:|
| symmetric outer boundary | 4.123 px | 13.073 px |
| predicted shoreline → GT outer boundary | 5.000 px | 13.732 px |
| GT outer boundary → predicted shoreline | 3.000 px | 12.908 px |

128 个采样 prediction 岸线点中，仅 40 个（31.25%）位于 GT 外岸线 2 px 内，
67 个（52.34%）位于 5 px 内。因此当前轮廓没有可靠贴合物理岸线。

## 4. 水位评价与误差解释

- estimated water level：-0.3650267410 m；
- GT water level：-0.3938947141 m；
- signed error：+0.0288679732 m；
- absolute error：2.8868 cm（约为 nominal 5 cm 的 57.74%）。

GT 水位只位于 prediction 岸线高程样本的第 16.41 百分位。128 个预测岸线射线中，
21 个落在 GT 水域内，其岸线高程中位数 -0.397029 m，接近 GT；107 个落在 GT
非水域，其岸线高程中位数 -0.358558 m，相对 GT 平均偏高 4.946 cm。错误外扩岸线
因此直接将 robust median 水位推高。

作为 evaluation-only 反事实检查，将严格 GT Camera 外岸线送入同一 ray–DEM 与
水位估计几何，得到 -0.3934653373 m，绝对误差仅 0.000429 m；原 prediction 的
ray–DEM residual P95 也只有 9.29e-9 m。证据表明相机几何和数值求交不是主要误差，
水位偏差可由错误岸线位置解释。

## 5. DEM mask、depth、面积和体积

DEM prediction 包含 257 cells，GT 为 158 cells，158 个 GT cells 全部被覆盖：

- DEM mask IoU / precision：0.614786；
- DEM mask recall：1.000000；
- false positive：99 cells；false negative：0 cells。

在已保存 prediction 水位下，重算的低地区域与已保存 DEM mask 完全一致（IoU 1.0），
说明给定该错误水位后，DEM 重建没有额外结构错误。depth 采用
`prediction - GT`，必须按评价域解释：

| 评价域 | cells | MAE | RMSE | bias |
|---|---:|---:|---:|---:|
| 全部有效 DEM | 12,600 | 0.000472 m | 0.003547 m | +0.000472 m |
| prediction 水域 | 257 | 0.023157 m | 0.024837 m | +0.023157 m |
| GT 水域 | 158 | 0.028868 m | 0.028868 m | +0.028868 m |
| overlap | 158 | 0.028868 m | 0.028868 m | +0.028868 m |

全 DEM 指标会被大量共同为 0 的干地稀释，不能作为水域 depth 精度结论。GT/overlap
域中误差几乎是固定的 +2.887 cm，与水位高估完全一致。

| 数量 | prediction | GT | 绝对误差 | 相对误差 |
|---|---:|---:|---:|---:|
| area | 2.57 m² | 1.58 m² | 0.99 m² | 62.66% |
| volume | 0.0990605 m³ | 0.0395474 m³ | 0.0595131 m³ | 150.49% |

mean / median / max depth 分别高估 1.3515 / 1.3764 / 2.8868 cm。

## 6. 误差来源结论

`dominant_error_source = mixed`，具体是：

- `segmentation_scope = true`：mask 范围明显过扩；
- `shoreline_localization = true`：外岸线 P95 为 13.073 px；
- `water_level_estimation = true`：错误岸线样本使水位高估 2.887 cm；
- `camera_geometry = false`：GT 岸线反事实和 ray residual 均支持几何正确；
- `dem_reconstruction = false`：给定 prediction 水位后重建与保存结果一致。

manual prompt mask 更接近“覆盖了完整水域、同时扩展到涟漪/反射和普通路面区域的候选”，
而不是精确的完整积水物理边界。改进优先级应为：先收紧 SAM 提示或候选语义范围，
再明确以物理外岸线而非涟漪外缘作为 shoreline；现有相机标定、ray–DEM 求交和
DEM flood reconstruction 暂无修改依据。

Phase 2D-C-3C prediction-side gate 为 `reject`，原因包括重投影 IoU、边界误差和岸线
高程离散。独立 GT 评价确认该 reject 合理，rejected candidate 不得进入 S5-S8。

## 7. 可重复性与非修改保证

评价脚本在运行前后记录 Phase 2D-C-3C 输出目录全部文件 SHA-256，结果一致。
本阶段测试覆盖二值 mask、双向边界、外岸线孔洞排除、depth 评价域、面积体积、
输入不变性、确定性及 prediction/GT 依赖隔离。所有自动生成评价结果保存在被
`.gitignore` 排除的 `outputs/sam2_shoreline_geometry_gt_evaluation/`。
