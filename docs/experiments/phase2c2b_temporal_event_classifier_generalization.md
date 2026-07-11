# Phase 2C-2B：时域事件分类器跨 seed 泛化

## 1. 背景与目标

Phase 2C-2A 的规则基线在 18 条有水序列上达到平均 Camera mask IoU 0.7149，但旧评价中的 water-event F1 仅 0.0863，6 条 dry 序列累计 322 条 false water tracks。最终 dry mask 为零主要依赖保守 evidence threshold 和 gate，而非事件分类已经可靠。

本阶段保留原规则分类器，新增 NumPy 实现的 L2 logistic regression，对 RGB track 手工特征做离线监督训练，并比较 rule、model 和 hybrid。当前仍为 synthetic-domain research output，固定 `eligible_for_downstream: false`。

## 2. Sequence-level split

禁止按帧或 track 随机拆分。最终 split 按完整 sequence 和 seed 固定：

| split | seeds | sequences | case 覆盖 | rain 分布 |
|---|---|---:|---|---|
| train | 41, 42, 43, 101, 102 | 18 | dry、5/10/20/40cm | light 5、moderate 8、heavy 5 |
| validation | 111, 112 | 6 | dry×2、5/10/20/40cm 各1 | light/moderate/heavy 各2 |
| test | 201, 202, 203 | 6 | dry×2、5/10/20/40cm 各1 | light/moderate/heavy 各2 |

seed overlap 为空，自动检查通过。validation/test 的补充序列为 5 秒、20 fps。配置和阈值只依据 train/validation 冻结。

## 3. 防泄漏结构

结构分为：

- `track_feature_extractor`：仅从 RGB-derived track 和曝光残差计算特征；
- `training_label_loader`：仅在离线训练读取 `ground_truth/event_annotations.json`；
- `classifier_trainer`：接收数值特征和匹配标签；
- `inference_classifier`：只接收 track features、冻结模型、阈值和规则配置；
- `evaluation_loader`：prediction 和 gate 写出后才读取 GT mask/event。

推理的外部文件输入只有 `frames/*.png`、detector 配置、`model_weights.npz`、feature normalization/schema 和冻结阈值。推理不读取 metadata、case、水深、rain level、seed、真实 mask 或 event type。模型权重、标准化和 feature schema 文件不包含 water mask、case depth 或 generator seed；seed 仅存在独立 split manifest 中。

自动测试覆盖删除/修改答案后模型推理不变的接口约束、函数签名无 GT 参数、模型文件禁用字段和训练确定性。

审计说明：开发早期训练脚本曾在同一命令末尾产生一次临时 test report。该报告未参与阈值选择，随后脚本被改为训练阶段完全不装载 test，临时结果被覆盖。最终正式流程为 train/validation 冻结模型后，由独立评价脚本首次生成最终 test artifacts；后续没有根据正式 test 修改模型权重或阈值。hybrid 的规则保底策略属于预先定义的安全组合语义，不修改 model 独立结果。

## 4. Track 与 GT 事件匹配

匹配器综合时间交集、temporal IoU、事件生命周期覆盖比例、track center 与事件中心距离，以及 bbox 对事件半径区域的空间关系。候选按综合分数全局排序并一对一分配；同一 track 的前两候选分差过小时标记 uncertain；其余未匹配 track 显式标记 `background_noise`。

| split | matched | dry | water | unmatched/noise | ambiguous | unmatched events | mean center error |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 727 | 452 | 275 | 44,712 | 407 | 8 | 5.03 px |
| validation | 136 | 94 | 42 | 7,984 | 54 | 0 | 4.73 px |
| test | 136 | 81 | 55 | 7,875 | 86 | 0 | 5.43 px |

大量 unmatched track 来自雨丝、噪声和碎片化组件，是主要类别不平衡来源。

## 5. 特征与标准化

复用 duration、peak、maximum area、area/radius growth、monotonicity、decay、post-peak persistence、drift、compactness、ringness、polarity changes、temporal energy/asymmetry，并新增：

- early radius growth；
- post-peak area integral；
- ring energy duration；
- radius expansion consistency；
- polarity switch rate；
- smoothed area/radius slope；
- observation fill ratio；
- local dynamic density；
- neighboring track count；
- exposure residual correlation。

所有特征先检查 finite，非 finite 值置零并计数。均值和标准差只由 train split 计算并保存；validation/test 只应用训练参数，绝不重新标准化。推理同时计算最大 train-z 供 OOD gate 使用。

## 6. 模型、类别不平衡与阈值

分类器是确定性二分类 logistic regression：NumPy batch gradient descent、L2=0.02、1800 iterations。water class weight 为 8，dry/noise 为 1；保留全部 water，单 sequence 负样本最多 300，negative:positive 目标为 5:1，并按 sequence 和 rain-level 反频率加权。

最终训练使用 1,980 条 track：water 275、dry match 422、background/noise 1,283。

validation-only 网格选择结果：

- `low_threshold = 0.45`
- `high_threshold = 0.80`
- 中间区间输出 uncertain

Validation water F1 为 0.3226（precision 0.2941、recall 0.3571），macro F1 为 0.6440。Test 没有参与阈值选择。

## 7. Model evidence 与 conservative hybrid

Model evidence 以 `model_water_probability`、生命周期和有限 Gaussian kernel 加权；uncertain 不作为 water seed，dry probability 形成有限负证据，并限制单 track 最大贡献。无支持区域继续为 unknown，不使用静态水面颜色或 GT 补全。

初始概率混合在 held-out test 明显抹掉了 rule mask。最终 hybrid 因而采用明确的安全回退：保留 Phase 2C-2A rule mask，model 只增加 probability/evidence corroboration 和 disagreement 诊断，不能删除规则 mask。该策略使 hybrid mask 与 rule mask 相同；这不是模型性能提升，而是防止尚未泛化的实验模型破坏已验证基线。

## 8. Held-out test 事件结果

下表使用新的稳健 matcher，因此与 Phase 2C-2A 旧的 0.0863 不是完全同一评价口径。

| 模式 | water precision | water recall | water F1 | dry F1 | macro F1 | background/noise false-water |
|---|---:|---:|---:|---:|---:|---:|
| rule baseline | 0.0323 | 0.4182 | 0.0600 | 0.0364 | 0.0482 | 638 |
| learned model | 0.0536 | 0.0545 | 0.0541 | 0.0206 | 0.0373 | 41 |
| hybrid classifier | 0.0909 | 0.0545 | 0.0682 | 0.0206 | 0.0444 | 22 |

模型显著降低 background/noise false-water，但 water recall 下降严重。Hybrid water F1 比同口径 rule 基线略高，却仍低于方向性目标 0.0863。不能宣称事件泛化问题已解决。

两条 held-out dry sequence 中，false-water tracks 为 rule 47、model 0、hybrid 0；三种模式的最终 dry water-mask false-positive pixels 均为 0。

## 9. Held-out Camera mask 结果

四条有水 test sequence 的平均值：

| 模式 | IoU | precision | recall | pixel F1 | boundary F1 | dry FP pixels |
|---|---:|---:|---:|---:|---:|---:|
| rule baseline | 0.6254 | 0.9929 | 0.6287 | 0.7420 | 0.3171 | 0 |
| learned model | 0.1817 | 0.9734 | 0.1842 | 0.2841 | 0.0503 | 0 |
| conservative hybrid | 0.6254 | 0.9929 | 0.6287 | 0.7420 | 0.3171 | 0 |

逐水位 IoU：

| case | rule | model | hybrid |
|---|---:|---:|---:|
| 5cm light seed201 | 0.2714 | 0.0108 | 0.2714 |
| 10cm heavy seed202 | 0.6363 | 0.2878 | 0.6363 |
| 20cm moderate seed203 | 0.6729 | 0.0859 | 0.6729 |
| 40cm heavy seed202 | 0.9209 | 0.3421 | 0.9209 |

按雨强、包含对应 dry sequence 的平均 IoU：light rule/model/hybrid 为 0.1357/0.0054/0.1357；moderate 为 0.3364/0.0430/0.3364；heavy 为 0.7786/0.3149/0.7786。

## 10. 消融实验

Validation feature ablation：

| 特征设置 | water F1 | macro F1 |
|---|---:|---:|
| all features | 0.3226 | 0.6440 |
| without ringness | 0.3226 | 0.6439 |
| without duration | 0.3077 | 0.6397 |
| without expansion | 0.3373 | 0.6527 |
| without post-peak persistence | 0.2909 | 0.6281 |
| without class weights | 0.3077 | 0.6521 |

Duration 和 post-peak persistence 有正向贡献；去掉 expansion 反而略升，说明当前 track fragmentation 使扩张特征不稳定。Ringness 贡献接近零。

Held-out temporal ablation 的 model mask IoU（包含 dry）为：full 0.1211、single frame 0、shuffled 0.3091、color-normalized 0.1127。Single frame 为零说明不是简单静态单帧分类，但 shuffled 明显高于 full，表明模型可能依赖顺序不敏感的动态密度、雨丝碎片或合成渲染统计捷径，尚不能证明主要学习到真实涟漪扩张规律。

## 11. Prediction-side quality gate

Model gate 不读取 GT，检查 confidence/uncertain 分布、water spatial clustering、probability separation、train-z OOD、非 finite 特征、动态饱和、单 track 贡献、时间窗口一致性、rule/model disagreement 和异常面积。

Test gate 分布：3 pass、3 partial、0 reject：

- 两条 dry：因时间窗口不一致和空 model mask 为 partial；
- 5cm light：因单 track evidence 主导为 partial；
- 10cm heavy、20cm moderate、40cm heavy：pass。

即使 pass，仍固定 `eligible_for_downstream: false`。鉴于 shuffled 消融异常，当前 gate 不能被视为已经完成真实域可靠性校准。

## 12. 结论与真实视频计划

本阶段成功实现了严格 sequence split、无泄漏训练/推理分离、确定性轻量模型、概率输出、OOD/gate 和完整 rule/model/hybrid 对比。模型将 held-out background/noise false-water 从 638 降到 41，dry false-water tracks 从 47 降到 0。

但核心目标只部分达到：water-event recall 和 model mask IoU 明显低于规则基线，乱序消融异常，扩张特征没有形成稳定优势。当前最佳安全选择是保留 rule mask，model 仅作研究诊断和未来 corroboration，禁止进入 Phase 2B 或 S5-S8。

真实验证应采集固定机位的 dry/wet/shallow-water、light/moderate/heavy、日间/夜间连续视频，按完整地点和日期拆分；人工标注 impact events 与 water region；重新检查 tracker fragmentation、扩张趋势和概率校准。若线性模型在真实域仍无法获得稳定 recall，再考虑带强时域归纳偏置的轻量序列模型，而不是直接训练静态 CNN。
