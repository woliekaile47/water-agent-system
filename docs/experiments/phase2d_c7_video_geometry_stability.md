# Phase 2D-C-7-3：连续帧岸线与几何稳定性诊断

## 目标与协议

本阶段只处理 Phase 2D-C-7-1 已冻结的 123 个 SAM 2 视频 mask，验证主体外岸线、ray–DEM 水位、Camera 可见面积/体积和深度的时间稳定性。

- 没有重新运行 SAM 2；
- 没有修改自动提示或传播 mask；
- 没有读取 Camera mask、水位、DEM mask、depth、area 或 volume GT；
- 使用同一个 dry Ground DEM、相机参数和现有 Phase 2B/C3C 配置；
- 复用现有 ray–DEM、MAD 过滤、seed basin、深度反演、重投影和 quality gate；
- 没有修改 3 px boundary gate 或其他阈值；
- 输出不是 authoritative measurement，不允许进入 S5-S8。

每帧仅保留最大连通域，使用 `RETR_EXTERNAL + CHAIN_APPROX_NONE` 提取主体外岸线，并按累计弧长固定采样 128 点。内部孔洞和零散小组件不参与水位岸线；未做填孔、平滑、膨胀、腐蚀或按 case 调参。

## 时间稳定性结果

| 指标 | 5 cm heavy | 20 cm moderate | 40 cm light |
|---|---:|---:|---:|
| 可用几何帧 | 41/41 | 41/41 | 41/41 |
| 水位范围（m） | -0.39490 ～ -0.36266 | -0.24725 ～ -0.23935 | -0.05487 ～ -0.04829 |
| 水位标准差（cm） | 0.860 | 0.197 | 0.142 |
| 相邻水位变化 P95（cm） | 1.672 | 0.337 | 0.341 |
| 相邻水位变化最大值（cm） | 2.021 | 0.365 | 0.569 |
| 面积中位数（m²） | 1.81 | 7.09 | 17.96 |
| 面积 CV | 15.23% | 0.87% | 0.60% |
| 体积中位数（m³） | 0.05093 | 0.67841 | 2.96577 |
| 体积 CV | 31.32% | 2.05% | 0.86% |
| 最大深度中位数（cm） | 5.675 | 20.095 | 39.371 |
| Camera 重投影 IoU 中位数 | 0.8475 | 0.9554 | 0.9639 |
| Boundary P95 中位数（px） | 4.183 | 2.236 | 3.162 |

这些是 prediction-side 时间稳定性，不是绝对准确度。尤其不能把 20/40 cm 的稳定数值直接当作真实水位或完整道路总面积。

## 5 cm heavy

水位全窗口跨度约 3.224 cm，面积 CV 15.23%，体积 CV 31.32%。40/41 帧因 Camera 重投影 IoU 低于现有阈值被拒绝，33 帧同时触发 boundary P95，2 帧触发岸线高程 MAD。

Anchor frame 99 的预测侧结果为：

- estimated water level：-0.370326 m；
- area：2.36 m²；
- volume：0.08611 m³；
- max depth：7.357 cm；
- Camera reprojection IoU：0.7357；
- boundary P95：8.378 px；
- gate：reject。

结合 C7-2 Camera GT 评价，主要问题仍是浅水强雨下的视觉候选过分割。时序传播和 DEM 几何无法从被污染的岸线恢复可靠语义。

## 20 cm moderate

水位标准差仅 0.197 cm，全窗口跨度约 0.790 cm；面积 CV 0.87%，体积 CV 2.05%。31/41 帧通过现有 prediction-side gate，10 帧仅因 boundary P95 超过 3 px 被拒绝。

Anchor frame 99：

- estimated water level：-0.241053 m；
- area：7.14 m²；
- volume：0.69189 m³；
- max depth：20.284 cm；
- Camera reprojection IoU：0.9541；
- boundary P95：2.828 px；
- gate：pass。

该序列证明连续帧几何链路可以稳定工作，但仍需独立 GT 评价后才能讨论绝对误差。

## 40 cm light

水位标准差 0.142 cm，全窗口跨度约 0.659 cm；面积 CV 0.60%，体积 CV 0.86%。虽然 Camera 可见候选十分稳定，41/41 帧的全局 gate 仍为 reject：

- 41 帧均存在 `ambiguous_candidate_basin`；
- 41 帧均存在 `candidate_basin_outside_camera_coverage`；
- 24 帧还触发 `boundary_reprojection_error_above_threshold`。

其中 17/41 帧满足 Camera 可见区域的其他条件，可标记：

- `observable_region_result_valid=true`；
- `global_estimate_status=partial`；
- `area_volume_semantics=observable_lower_bound`。

剩余 24 帧因 boundary P95 超过 3 px 被标为 `unavailable`。系统没有自动加入无 Camera seed 的第二盆地，保持了不可观测盆地安全性。

## 与约 3 cm 水位目标的关系

20 cm 和 40 cm 序列的相邻水位变化 P95 分别只有约 0.34 cm，明显小于 3 cm；5 cm heavy 的相邻变化 P95 为 1.67 cm，最大约 2.02 cm，但其全窗口跨度略高于 3 cm，且 Camera 语义与面积/体积明显不稳定。

这说明：

1. 对中深水可见主盆地，时间稳定性已不是主要瓶颈；
2. 对浅水强雨，单看水位变化小于 3 cm 仍不足以判定有效，因为错误岸线也可产生看似平滑的水位；
3. 3 px 像素阈值与 3 cm 水位目标不能直接等同；
4. C8 应根据独立 GT 标定岸线像素误差、水位误差和时间稳定性的联合关系，而不是单独放宽像素阈值。

## Quality gate 迁移证据

本阶段支持在 C8 中明确区分：

- `camera_visible_estimate`：Camera 有证据且几何/时序可信的可见主盆地估计；
- `global_scene_estimate`：只有 Camera 覆盖全部候选盆地且不存在歧义时才允许成立。

建议的多指标 gate 输入包括：

- Camera reprojection IoU；
- 主体外岸线 P95，而非孔洞或碎片边界；
- 岸线高程 MAD/IQR；
- 连续帧 water-level 标准差与相邻变化 P95；
- 面积/体积 CV；
- 不可观测 candidate basin 数量；
- Camera mask 语义置信和浅水强雨状态。

本阶段只提供证据，没有修改正式 gate。

## 输出与结论

输出目录：

`outputs/sam2_video_geometry_stability_seed302_frame79_119/`

共处理 123 帧，全部几何计算可用，运行耗时约 429.4 秒。

结论：20 cm moderate 和 40 cm light 的 Camera 可见水位、面积与体积具有良好时间稳定性；5 cm heavy 仍不可靠。下一步应进行独立的连续几何 GT 评价，将预测水位、面积、体积和深度与静态同场景 GT 对齐，然后进入 C8 gate 重构。
