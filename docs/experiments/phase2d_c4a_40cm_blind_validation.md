# Phase 2D-C-4A：40 cm SAM 2 盲测验证

## 实验目的

本实验验证人工提示 SAM 2 可见水域候选在严格盲测条件下，能否通过既有
Camera岸线、dry Ground DEM和Phase 2B几何链路得到可审计的水位与水深结果。
实验只评价单Camera可观测范围，不声称已经实现整段道路的完整全局积水测量，
结果不得进入正式S5-S8。

## 严格盲测流程

1. 预先固定 `sim_water_40cm_001/heavy/seed_43` 的第59帧RGB；
2. 在不查看任何40 cm Ground Truth的情况下完成人工提示；
3. 固定最高分SAM 2 Candidate 1及其raw mask，不使用GT重选候选或调整提示；
4. 从raw mask提取主体外岸线，冻结prediction-side水位、basin、depth、面积、体积和gate状态；
5. 记录prediction文件哈希后，最后由独立evaluation模块读取严格同场景GT；
6. evaluation没有修改或重新运行prediction，没有使用GT重选水位、seed或basin。

## 冻结输入与追溯

- case：`sim_water_40cm_001`
- sequence：`sim_water_40cm_001/heavy/seed_43`
- frame：`frame_000059.png`，640×360
- RGB SHA-256：`a900f45759d31971a6843e4129078576df7f8ef29cccba446e2eb753523d2495`
- selected mask SHA-256：`2b6fc873459dbce2e2adbf88759e42962c766de6119c0b3d6149d0655fe0e17e`
- shoreline JSON SHA-256：`63518cf0eacfde43122edbf67518bbd5652d9ea96cb3381bf2d97822dc330084`
- prediction主结果SHA-256：`d0e31be3a96ee6274eb80bfe5600e88afeafc836c96a7bd72e45a06ef69056e6`

## SAM 2 Camera mask

| 指标 | 结果 |
|---|---:|
| predicted / GT pixels | 20,991 / 21,656 |
| IoU | 0.944244 |
| precision | 0.986709 |
| recall | 0.956409 |
| F1 | 0.971323 |
| false positive / false negative | 279 / 944 pixels |

主体外岸线的symmetric P50/P95为1.000/3.162 px。prediction→GT P95为
2.906 px，GT→prediction P95为3.606 px。相比5 cm实验的Camera IoU
0.613886，本次40 cm可见水域候选明显更准确。

## 水位与水深

- prediction water level：`-0.0450970985 m`
- GT water level：`-0.0438947141 m`
- 水位绝对误差：`0.12024 cm`
- prediction最大水深：`39.8798 cm`
- GT最大水深：`40.0000 cm`
- 最大水深误差：`-0.1202 cm`

严格GT岸线经过同一ray–DEM链路得到的水位绝对误差为0.08693 cm，支持当前
Camera几何与数值求交不是主要误差来源。

## DEM、面积和体积

DEM mask的IoU/precision/recall/F1分别为
0.829941/1.000000/0.829941/0.907069。预测没有误扩张到GT外，但全局漏掉376个
GT水域栅格。

| 数量 | prediction | GT | 误差 |
|---|---:|---:|---:|
| area | 18.35 m² | 22.11 m² | 低估3.76 m²，17.01% |
| volume | 3.058095508 m³ | 3.095836190 m³ | 低估0.037740682 m³，1.22% |

面积不是完整全局估计。体积误差较小，是因为未选的第二盆地较浅，不能据此把
18.35 m²解释为整段道路总积水面积。

## 第二盆地与Camera可观测性

预测水位下共有两个候选盆地：

- 主盆地：1,835 cells、18.35 m²，包含34个有效Camera seed，全部属于GT水域；
- 第二盆地：300 cells、3.00 m²，没有Camera seed，Camera投影像素为0，
  独立GT评价确认300/300 cells均为真实水域。

现有投影只能确认第二盆地没有有效Camera投影，不能进一步区分图像外与遮挡。
prediction保持seed-connected策略，没有利用GT或DEM低洼先验自动补入该盆地。
这是安全但保守的可观测范围估计，也是全局面积不足的主要原因。

## Quality gate

冻结gate结果为`reject / diagnostic_only`，原因是：

- `boundary_reprojection_error_above_threshold`；
- `ambiguous_candidate_basin`；
- `candidate_basin_outside_camera_coverage`。

完整重投影boundary P95为40.311 px，候选mask含72个内部孔洞、575个孔洞像素和
1,083个孔洞相邻边界像素；只看主体outer boundary时P95为3.097 px，仍略高于
原3.0 px阈值。本实验没有修改阈值或把outer boundary替换为正式gate输入。

## 结论

- `prediction_accuracy_status = partially_accurate`；
- `dominant_error_source = conservative_quality_gate`；
- 主Camera可见水域、岸线水位、最大水深和体积预测准确；
- 单Camera无法提供第二盆地证据，因此全局面积不完整；
- reject对正式下游链路是合理的保守决策；
- 当前结果保持`authoritative=false`和`eligible_for_downstream=false`，不接入S5-S8；
- 值得继续进行20 cm和10 cm严格盲测，检查分割精度、水位误差和可观测性边界随水深的变化。
