# Phase 2D-B-2B 岸线拓扑修复离线对照实验

## 1. 实验目标与约束

本实验只对 Phase 2D-A 已有最终 `predicted_camera_water_mask.png` 做固定、可解释的离线拓扑处理，然后复用冻结的 Camera-mask-to-DEM 几何链路，判断是否能降低 boundary reprojection P95，同时避免面积膨胀、Camera IoU 下降和不可观测盆地填充。

实验严格遵守：

- 未启动 ROS、Gazebo、Camera、LiDAR 或 RTSP；
- 未重新运行 temporal prediction；
- 未修改 Phase 2D-A 正式 prediction 核心；
- 未修改 gate 状态机或任何阈值；
- 未修改 GT 或正式评价域；
- 方法和参数在 GT 读取前由统一配置固定；
- 未按 case 或 GT 指标选择最佳方法；
- 未生成 oracle 方法作为后续配置；
- 所有候选均为离线诊断，不是 authoritative measurement；
- 未接入 S5–S8、Agent、数据库、FastAPI 或 Dashboard；
- `eligible_for_downstream=false`。

## 2. 数据与隔离流程

实验对象：

- 23 条 water sequence；
- 10 条 dry sequence；
- water 每条执行 9 个固定候选，共 207 条评价结果；
- dry 每条执行 9 个变换，共 90 条空 mask 安全检查；
- 6 条 40 cm sequence × 9 方法，共 54 条不可见盆地安全记录。

每条 water sequence 严格执行：

```text
load existing predicted water / unknown masks
→ apply all nine fixed topology methods
→ complete shoreline, ray–DEM, water-level, DEM reconstruction and gate for all methods
→ close prediction-side candidate phase
→ independent evaluation loads Camera/DEM GT
→ compute IoU, water-level error, depth MAE and 40 cm basin safety
```

GT loader 只在全部固定候选完成后的独立评价函数中局部导入。Prediction-side candidate 函数不导入 GT loader。

Baseline 轻量重算的 23 条 P95 与 Phase 2D-B-1 已有 P95 完全一致，最大绝对差为 0 px，说明几何复用没有改变正式 P95 定义。

## 3. 固定方法和参数

统一配置文件为 `configs/shoreline_topology_ablation.yaml`。

| 方法 | 固定处理 |
|---|---|
| baseline | 原 mask，不处理 |
| largest_component | 只保留最大连通域 |
| small_component_filter | 删除面积 <100 px 的组件 |
| conditional_hole_fill | 只填充面积 ≤500 px 且 ≤原 water mask 面积 2% 的封闭孔洞 |
| morphological_closing | 固定 5×5 椭圆 kernel、1 次 closing |
| outer_boundary_only | mask 不变；水位岸线只使用最大主体外轮廓 |
| largest_component_conditional_hole_fill | 最大组件后条件填洞 |
| small_component_filter_conditional_hole_fill | 小组件过滤后条件填洞 |
| largest_component_outer_boundary_only | 最大组件后，水位只用主体外轮廓 |

没有方法使用 per-case 参数。

`outer_boundary_only` 只改变水位岸线提取，不修改 Camera water mask。现有 self-consistency gate 仍以完整 repaired mask 计算边界 P95，因此该方法不会通过偷偷改变评价域来降低 P95。

## 4. 全数据集方法汇总

下表中 P95 change 定义为 `after - before`，负值表示改善。

| 方法 | P95平均变化 | 改善/恶化/不变 | Camera IoU平均变化 | Camera IoU下降数 | 面积平均变化 | 面积范围 | Existing gate pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.000 px | 0/0/23 | 0.000000 | 0 | 0.000% | 0% | 0 |
| largest_component | +5.725 px | 2/3/18 | -0.027776 | 3 | -5.690% | -55.484%–0% | 0 |
| small_component_filter | -0.355 px | 3/0/20 | -0.000286 | 2 | -0.220% | -2.798%–0% | 0 |
| conditional_hole_fill | -0.023 px | 1/0/22 | +0.000049 | 0 | +0.005% | 0%–+0.114% | 0 |
| morphological_closing | -0.034 px | 3/4/16 | +0.000452 | 2 | +0.102% | 0%–+0.678% | 0 |
| outer_boundary_only | -0.033 px | 5/5/13 | 0.000000 | 0 | 0.000% | 0% | 0 |
| largest_component + hole fill | +5.702 px | 3/3/17 | -0.027728 | 3 | -5.685% | -55.484%–+0.114% | 0 |
| small filter + hole fill | **-0.378 px** | **4/0/19** | -0.000237 | 2 | -0.215% | -2.798%–+0.114% | 0 |
| largest component + outer only | +5.661 px | 6/5/12 | -0.027776 | 3 | -5.690% | -55.484%–0% | 0 |

结论：`small_component_filter + conditional_hole_fill` 是本批唯一同时满足“P95 改善数最多之一、P95 恶化数为 0、面积变化较小”的固定候选，但 19/23 不变、没有任何 sequence 通过现有 geometry gate，因此其改善仍然有限。

## 5. 面积膨胀与 Camera IoU 副作用

没有固定方法造成大规模面积膨胀：

- closing 最大面积增加 0.678%；
- conditional hole fill 最大增加 0.114%；
- small filter + hole fill 的最大增加同样为 0.114%。

主要危险不是膨胀，而是 `largest_component` 的过度删除：

- 最大面积缩减 55.484%；
- 20 cm light/41 删除第二个大组件，面积缩减 34.165%；
- 40 cm heavy/204 面积缩减 35.324%；
- 40 cm light/41 面积缩减 55.484%。

P95 改善但 Camera IoU 下降的案例集中在小组件过滤：

| Case | 方法 | P95 before→after | Camera IoU before→after | 面积变化 |
|---|---|---:|---:|---:|
| 40 cm heavy/204 | small filter（及其 hole-fill 组合） | 37.828→31.694 | 0.7422→0.7361 | -0.825% |
| 40 cm light/41 | small filter（及其 hole-fill 组合） | 31.006→30.942 | 0.4290→0.4170 | -2.798% |

这些结果说明小组件中既有错误碎片，也可能包含真实 Camera water evidence；不能仅凭 P95 改善无条件删除。

## 6. 五个极端 P95 案例

### 6.1 40 cm heavy/204

Baseline：4 个组件，P95 37.828 px，Camera IoU 0.7422。

- small component filter：组件降为 2，P95 降至 31.694 px，但 Camera IoU 降至 0.7361；
- closing：P95 不变为 37.828 px；
- conditional hole fill：无变化；
- outer boundary only：仅降至 37.815 px；
- largest component：P95 恶化到 96.153 px，Camera IoU 降至 0.4810。

这说明该案例的主要问题不是封闭孔洞，而是多个大区域、开口凹陷和碎片拓扑。单纯保留最大组件会删除大量有效区域，明显恶化。

### 6.2 40 cm light/41

Baseline：10 个组件，P95 31.006 px，Camera IoU 0.4290。

- closing：P95 改善至 29.428 px，Camera IoU 小幅升至 0.4319，面积增加 0.678%；
- small filter：P95 仅降至 30.942 px，Camera IoU 降至 0.4170；
- largest component：P95 恶化至 91.089 px，Camera IoU 降至 0.1910；
- outer boundary only：P95 恶化至 31.180 px。

closing 对该碎片化案例改善最大，但仍远高于 3 px，且不能推广为稳定方法，因为全数据集有 4 条 P95 恶化。

### 6.3 20 cm light/41

该 mask 有两个大组件。largest component 删除第二大水域后：

- 面积下降 34.165%；
- Camera IoU 从 0.5355 降至 0.3526；
- P95 从 14.879 恶化至 35.222 px。

因此 largest component 不能把“非最大组件”自动视为错误碎片。

### 6.4 10 cm heavy/202 与 20 cm moderate/203

两条均为单组件局部形状误差：

- 10 cm heavy/202：多数 mask-only 处理无效；outer boundary only 将 P95 从 14.204 降至 13.660，但 Camera reprojection IoU 同时下降；
- 20 cm moderate/203：closing 将 P95 从 13.441 小幅降至 13.322，其余方法基本不变。

固定拓扑处理不能解决这类水位或外轮廓形状偏差。

## 7. 近阈值五条

5 条 `3 < P95 ≤ 4` 案例在所有固定方法下：

- 平均 P95 change 均为 0；
- 没有方法造成 P95 恶化；
- 原 mask 均为干净单组件，固定方法基本没有改变它们。

因此保守的固定处理没有破坏近阈值案例，但也没有把它们降到 3 px 以下。近阈值问题仍更接近离散轮廓/水位投影差异，不是小组件或封闭孔洞问题。

## 8. 八条高 Camera IoU reject

改善只出现在两个 40 cm heavy 案例：

- heavy/43：conditional hole fill 将 P95 5.000→4.472，Camera IoU 0.9444→0.9455，面积 +0.114%；
- heavy/112：outer boundary only 将 P95 6.325→6.000，Camera mask 与面积不变。

其余高 IoU 案例没有稳定改善。所有方法在现有 gate 下仍为 reject。

## 9. 封闭孔洞与 outer-boundary-only

23 条 baseline mask 中只有 40 cm heavy/43 存在一个满足定义的真实封闭孔洞。B-2A 图上看似“内部孔洞”的其他极端结构，实际多为：

- 与外部背景连通的开口凹陷；
- 不同大组件之间的空隙；
- fragmented water evidence；
- unknown 或已知非水通道，而非封闭 hole component。

因此 conditional hole fill 只影响一条 sequence，并将其 P95 从 5.0 降至 4.472 px。内部封闭孔洞边界不是全体极端 P95 的主要来源。

`outer_boundary_only` 保持 Camera mask 和正式 P95 评价域不变，仅改变水位样本。它在 5 条改善、5 条恶化，平均变化仅 -0.033 px，不能作为稳定修复。结果说明仅忽略内部水位岸线并不足以修复最终重投影边界。

## 10. Dry 安全性

10 条 dry × 9 方法的 90 条检查全部满足：

```text
input water pixels = 0
output water pixels = 0
empty_preserved = true
```

所有固定方法均未从空 mask 产生 false water。

## 11. 40 cm 不可见盆地安全性

6 条 40 cm sequence × 9 方法，共 54 条安全记录：

- 每条均识别 1 个不可见第二盆地；
- 不可见第二盆地 Camera projected pixels 为 0；
- repaired Camera evidence overlap 最大值为 0；
- predicted DEM intersection 最大值为 0；
- `all_methods_preserve_unobservable_secondary_basin=true`。

没有方法生成第二盆地 Camera 证据，没有自动填充第二盆地，也没有把 DEM 低洼先验冒充 Camera prediction。

## 12. 是否存在稳定固定候选

存在一个“相对最稳定但作用有限”的候选：

```text
small_component_filter + conditional_hole_fill
```

支持理由：

- 4 条改善、0 条恶化、19 条不变；
- 全体平均 P95 改善 0.378 px；
- 平均 Camera IoU 仅下降 0.00024；
- 平均面积减少 0.215%；
- dry 安全；
- 40 cm 不可见盆地安全。

限制：

- 改善集中在少数有小组件/单个封闭孔洞的案例；
- 40 cm extreme 仍约为 31 px；
- 2 条 Camera IoU 下降；
- 23 条中没有任何一条通过现有 geometry gate；
- 未解决单组件局部形状、水位和 DEM 重建误差；
- 仅在 synthetic visual abstraction 上验证，尚无真实视频证据。

## 13. 是否进入正式 prediction 修改

当前证据不足以直接修改正式 prediction。

建议下一步先做受控的 Phase 2D-B-2C 候选验证：

1. 只把 `small_component_filter + conditional_hole_fill` 作为实验候选，不替换 baseline；
2. 对小组件增加跨帧持续性、空间邻接和水纹证据，而不是仅按面积删除；
3. 保存逐帧 masks/shorelines，验证被删组件是否真实短时伪影；
4. 针对单组件极端案例执行固定 Camera mask 下的 water-level sweep；
5. 在独立 synthetic holdout 和真实标注视频上验证后，再决定是否进入 prediction 核心；
6. 保持现有 3 px gate 不变，不能因拓扑候选仍 reject 而放宽阈值。

## 14. 输出

输出目录：

```text
outputs/synthetic_visual_to_depth_integration/shoreline_topology_ablation/
```

包含：

- `topology_ablation_results.csv`
- `topology_ablation_results.json`
- `method_summary.json`
- `dry_safety_summary.json`
- `forty_cm_basin_safety.json`
- `experiment_config_snapshot.json`
- `p95_before_after_by_method.png`
- `camera_iou_before_after_by_method.png`
- `area_change_by_method.png`
- `extreme_case_comparison.png`
- `near_threshold_case_comparison.png`

自动输出继续由现有 `outputs/synthetic_visual_to_depth_integration/` 忽略规则排除，不进入 Git。
