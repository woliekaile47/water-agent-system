# Phase 2A Camera Mask + Ground DEM 水深反演实验

## 1. 实验目的

Phase 2A 验证 Two-stage Perception 路线中的第一个离线算法闭环：不在降雨期间依赖 LiDAR 扫描水面，而是使用无水阶段保存的 Ground DEM 和降雨阶段的 Camera water mask，估计积水边界水位并反演水深、面积和体积。

本实验使用仿真 Camera Ground Truth mask 代替尚未开发完成的雨滴视觉模型，目的是先单独验证以下链路：

```text
dry Ground DEM
+ Camera water mask
+ CameraInfo / 外参
→ Camera mask 投影到 DEM
→ DEM-space 水域边界
→ 水面高程估计
→ Ground DEM 水深反演
→ 面积、体积
→ 独立 Ground Truth 评价
→ quality gate
```

本阶段结果仅用于仿真实验，`eligible_for_downstream` 始终为 `false`，不得进入正式 S5-S8 预警链路。

## 2. 数据输入

预测阶段使用以下输入：

- `data/simulation/sim_dry_baseline_001/ground_truth/ground_dem_gt.npy`：无水 Ground DEM；
- `data/simulation/<case_id>/ground_truth/camera_water_mask_gt.png`：Camera 空间的水域二值 mask；
- `simulation/config/sensors.yaml`：Camera 内参、Camera 位姿、坐标系、DEM 范围和分辨率；
- 统一的 Phase 2A 投影、边界反演和质量门控配置。

实验场景为：

- `sim_water_5cm_001`
- `sim_water_10cm_001`
- `sim_water_20cm_001`
- `sim_water_40cm_001`

自动生成的预测与评价结果保留在 `outputs/simulation_evaluation/`，供 Phase 2B 对比，但该目录不进入 Git。

## 3. Ground Truth 防泄漏设计

预测输入加载器和 Ground Truth 评价加载器在代码结构上分离。

预测阶段只允许读取 dry Ground DEM、Camera mask 和非答案性质的相机/场景几何配置，不读取：

- water case manifest 中的真实水位、面积或体积；
- `dem_water_mask_gt.npy`；
- `depth_map_gt_m.npy`；
- `ground_truth_metadata.json` 中的真实水位、水深、面积或体积。

预测阶段完成投影、水位反演、水深计算并生成 GT-independent quality gate 后，独立 evaluation loader 才读取上述 Ground Truth 答案计算评价指标。

自动测试使用损坏的 manifest、DEM GT mask 和 depth GT 文件验证预测链路仍能运行，证明预测函数不依赖这些答案文件。真实水位没有用于选择估计方法，也没有用于逐 case 调参。

## 4. Camera mask 到 DEM 的映射方法

使用“DEM 栅格反向投影到 Camera”的方法，不从单个图像像素直接猜测三维深度：

1. 根据 DEM shape、道路范围和分辨率构造每个栅格中心坐标；
2. 从 dry Ground DEM 读取栅格地面高程，形成 `P_map=(x,y,z_ground)`；
3. 使用 `T_camera_optical_map` 将点转换到 `camera_optical_frame`；
4. 光学坐标遵循 X 向右、Y 向下、Z 向前；
5. 过滤非有限 DEM、Camera 后方、near/far clip 以外及图像范围外的点；
6. 使用 CameraInfo 内参投影：`u=fx·X/Z+cx`、`v=fy·Y/Z+cy`；
7. 读取 Camera mask 对应像素，生成 predicted DEM water mask；
8. 记录投影有效率和所有无效原因。

四个场景的 DEM 有效投影率均为 `0.729603`。

## 5. 边界水位估计方法

统一配置采用 `inner_outer_bracket_midpoint`，所有 case 使用相同参数：

1. 对 predicted DEM mask 进行 8 邻域连通域分析；
2. 删除小面积噪声连通域；
3. 每个保留连通域分别提取 inner boundary 和一圈 outer ring；
4. 从 dry Ground DEM 读取内外边界高程；
5. 过滤 NaN、Inf，并使用 MAD 剔除离群值；
6. inner boundary 使用较高分位数，outer ring 使用较低分位数；
7. 内外边界能形成合理区间时取中点作为水面高程；
8. outer ring 无效或区间不合理时回退到 inner boundary 稳健估计。

得到水面高程后，仅在 predicted DEM mask 内计算：

```text
depth(x,y) = max(0, estimated_water_level - ground_dem(x,y))
```

mask 外有效 DEM 栅格强制为 0；无效 DEM 保留为 NaN；面积按有效水域栅格数积分，体积按 `sum(depth × cell_area)` 积分。

## 6. 5/10/20/40 cm 评价结果

| Case | IoU | Precision | Recall | Boundary F1 | 水位绝对误差 | Union depth MAE | Union depth RMSE | 面积绝对误差 | 体积相对误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 cm | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.111 mm | 0.111 mm | 0.111 mm | 0.00 m² | 0.442% |
| 10 cm | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.326 mm | 0.325 mm | 0.325 mm | 0.00 m² | 0.658% |
| 20 cm | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.874 mm | 0.874 mm | 0.874 mm | 0.00 m² | 0.921% |
| 40 cm | 0.8354 | 0.9995 | 0.8358 | 0.7976 | 0.116 mm | 0.803 mm | 2.153 mm | 3.62 m² | 0.574% |

以上结果来自统一配置的批量运行，没有使用真实水位选择每个场景的最佳参数。Depth MAE/RMSE 明确在 Ground Truth 与 prediction 的并集区域计算，不只统计预测正确区域。

## 7. Quality gate 结果

| Case | Gate | Reasons | Eligible for downstream |
|---|---|---|---|
| 5 cm | pass | `[]` | false |
| 10 cm | pass | `[]` | false |
| 20 cm | pass | `[]` | false |
| 40 cm | pass | `[]` | false |

当前 quality gate 检查 Camera mask 是否为空、投影有效率、DEM mask、连通域数量与占比、有效边界样本数、边界 MAD/IQR、内外边界 bracket、水面高程、物理水深上限、负水深、面积体积、Inf 和预测文件完整性。

Quality gate 不读取 Ground Truth。因此，它不会使用 IoU、recall、真实水位误差或真实面积误差决定 pass/reject；即使 gate 为 pass，本阶段结果仍被强制禁止进入 S5-S8。

## 8. 40 cm 场景限制

40 cm 场景的关键结果为：

```text
IoU = 0.8354
Recall = 0.8358
面积误差 = 3.62 m²
```

原因是当前算法把 **Ground DEM 点** 投影到 Camera mask。在较高水位下，实际 mask 描述的是水平水面投影，而预测查询点位于更低的地面高程，水面与地面之间产生明显视差。水位越高，视差越大，导致部分水域边缘无法由 ground-level 投影正确采样，主要表现为 recall 和预测面积下降。

当前 quality gate 没有利用 Ground Truth，因此无法通过 IoU 或 recall 发现这一问题。这是有意的防泄漏约束，不能通过把 Ground Truth 指标接入预测 gate 来规避。后续应增加不依赖 Ground Truth 的投影几何和边缘截断诊断。

## 9. 当前结论

- Phase 2A 已证明 Camera mask、无水 Ground DEM、CameraInfo 和外参可以形成可重复的边界水位与水深反演闭环；
- 5、10、20 cm 场景的 DEM mask、边界水位、面积和体积结果与仿真 Ground Truth 高度一致；
- 边界 bracket 方法能够在不读取真实水位的情况下估计水面高程；
- Ground Truth 预测/评价隔离已经由代码结构和自动测试保证；
- 40 cm 结果表明，单次 ground-level 反向投影在水位较高时存在系统性视差限制；
- 当前结果只适合作为 Phase 2B 算法开发基线，尚不具备进入正式预警链路的资格。

## 10. Phase 2B 改进方向

1. 设计水面高程与投影相互迭代的方法：先获得初始水位，再把候选水平水面重新投影到 Camera，更新 DEM mask 和边界；
2. 比较 ground-point projection、estimated-water-plane projection 和 camera-ray/DEM intersection 三种映射方式；
3. 增加不依赖 Ground Truth 的视差风险指标、图像边缘截断率、可见水域覆盖率和投影敏感度诊断；
4. 对 Camera mask 进行边界置信度建模，并将 mask 不确定性传播到水位、面积和体积区间；
5. 增加相机支架偏移、外参误差、夜间反射和不完整 mask 的鲁棒性测试；
6. 在仿真链路稳定后接入雨滴动态视觉 mask MVP，并保持视觉模型与 DEM 反演模块分开验收；
7. 使用真实视频和实测水尺进行独立验证，仿真指标不得替代真实场景验收。
