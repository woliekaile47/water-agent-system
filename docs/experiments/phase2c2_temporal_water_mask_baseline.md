# Phase 2C-2A：可解释时域积水 Camera mask 基线

## 1. 实验定位

本阶段建立一个不训练神经网络、仅依赖连续 RGB 帧的确定性基线。链路为：连续帧预处理 → 动态候选 → 时空轨迹 → dry splash / water ripple 规则分类 → 稀疏水面证据 → Camera water probability/mask → prediction-side quality gate → 独立 Ground Truth 评价。

当前结果属于合成域研究输出，不接入 Phase 2B、S5-S8、Agent 或正式预警链路；所有结果均保持 `eligible_for_downstream: false`。

## 2. 输入、输出与防泄漏边界

预测主函数只接受 `frames_dir`、检测器配置和消融模式。`detector_input_loader` 强制输入目录名为 `frames`，只按连续编号读取 `frame_*.png`，不遍历父目录。预测与 gate 完成后，独立的 `evaluation_ground_truth_loader` 才读取 `ground_truth/water_mask.png`、`event_map_sequence.npy` 和 `event_annotations.json`。

自动测试覆盖：损坏 GT/metadata 后预测不变、只复制 `frames/` 仍可运行，以及预测函数签名不接收真实 water mask。预测 manifest 和 gate 均记录 `ground_truth_used: false`；评价文件单独标记 `data_role: evaluation`。

每条序列输出 probability、water/unknown mask、evidence count、候选、轨迹、分类、诊断、gate、manifest、独立评价和对比图。自动结果位于 `outputs/temporal_water_detection/`，不进入 Git。

## 3. 帧预处理

RGB 转灰度后使用小尺度 Gaussian 去噪。每帧中位亮度与序列中位参考之差作为全局曝光偏移，并从整帧扣除；校正过程不读取 water mask。随后计算相邻帧 signed、absolute、positive 和 negative residual，并输出全序列最大残差预览。颜色均值归一化只作为消融分支。

## 4. 动态事件检测与跟踪

每帧阈值取固定残差阈值和帧内高分位阈值的较大值，经过有限闭运算/膨胀后做连通域分析。候选包含中心、bbox、面积、残差、极性、偏心率、紧致度、ringness 和等效半径。

候选通过中心距离、bbox IoU、时间间隔和面积比做确定性贪心关联，允许有限漏检。开发中发现并修复了一处坐标错误：连通域局部中心最初未加回 bbox 原点，曾导致轨迹错误聚集到左上角；回归测试现明确验证全图坐标。

## 5. 时域特征与规则分类

每条轨迹计算 duration、peak offset、最大面积、面积/半径增长斜率、扩张单调性、衰减率、峰后持续性、中心漂移、紧致度、ringness、极性变化、时域能量和不对称性。

统一规则将生命周期长、持续扩张、峰后能量持续、结构较环状且中心稳定的轨迹提高 `water_ripple_score`；短寿命、增长有限、快速衰减和结构不规则的轨迹提高 `dry_splash_score`。分数不足或 margin 太小的轨迹保持 `uncertain`。所有 case、雨强和 seed 使用同一配置。

## 6. 稀疏 water evidence 与 unknown

仅对分类为 water ripple 的轨迹，以事件中心为核、按置信度和生命周期加权累积有限半径 Gaussian evidence。概率由累计证据归一化得到，只做有限闭运算，不按水面颜色、case、水深或 Ground Truth 扩张。

从未获得事件证据的像素标记为 `unknown`，语义是 `no_temporal_evidence_not_confirmed_dry`，不能解释为 dry。该保守语义使预测通常具有较高 precision，但 light rain 下 recall 和覆盖率偏低。

## 7. Prediction-side quality gate

Gate 只使用帧数、fps、尺寸一致性、候选/轨迹数量、高置信 water 轨迹数、证据覆盖率、unknown 比例、mask 连通域、最大连通域比例、证据集中度、曝光异常、时序敏感性、时间稳定性、特征分离度、NaN/Inf 和异常面积等预测侧指标。即使 `pass`，仍固定 `eligible_for_downstream: false`。

24 条序列 gate 分布为 13 pass、11 partial、0 reject。按雨强：light 为 2 pass / 5 partial，moderate 为 6 pass / 3 partial，heavy 为 5 pass / 3 partial。light 常因证据不足而 partial；部分 heavy 因事件重叠、覆盖/稳定性问题而 partial，符合本阶段预期。

## 8. 四水位 × 三雨强结果

下表为 Phase 2C-1 原始 10 秒、20 fps、seed 41/42/43 序列。指标均由预测和 gate 完成后独立读取 GT 计算。

| 水位 | 雨强 | seed | IoU | precision | recall | gate |
|---|---|---|---:|---:|---:|---|
| 5cm | light | 41 | 0.5843 | 1.0000 | 0.5843 | pass |
| 5cm | moderate | 42 | 0.8911 | 0.9154 | 0.9710 | pass |
| 5cm | heavy | 43 | 0.6828 | 0.6852 | 0.9949 | pass |
| 10cm | light | 41 | 0.6431 | 1.0000 | 0.6431 | partial |
| 10cm | moderate | 42 | 0.8874 | 0.9811 | 0.9029 | pass |
| 10cm | heavy | 43 | 0.7891 | 0.7904 | 0.9978 | pass |
| 20cm | light | 41 | 0.5355 | 1.0000 | 0.5355 | pass |
| 20cm | moderate | 42 | 0.8934 | 0.9843 | 0.9064 | pass |
| 20cm | heavy | 43 | 0.8748 | 0.8770 | 0.9971 | pass |
| 40cm | light | 41 | 0.4290 | 1.0000 | 0.4290 | partial |
| 40cm | moderate | 42 | 0.9204 | 0.9972 | 0.9228 | pass |
| 40cm | heavy | 43 | 0.9444 | 0.9663 | 0.9765 | pass |

按水位汇总（包含额外 seed），平均 IoU 为：5cm 0.6074、10cm 0.7376、20cm 0.7441、40cm 0.7590。18 条有水序列整体平均 IoU 0.7149、precision 0.9539、recall 0.7592。

## 9. 额外 seed 泛化验证

额外序列统一缩短为 5 秒、20 fps，仅用于打破 seed 与 rain level 的绑定，未按 case 改配置。

| case | 雨强 | seed | IoU | precision | recall | water pixels | gate |
|---|---|---|---:|---:|---:|---:|---|
| dry | moderate | 101 | 0.0000 | 0.0000 | 0.0000 | 0 | partial |
| dry | light | 205 | 0.0000 | 0.0000 | 0.0000 | 0 | partial |
| dry | heavy | 206 | 0.0000 | 0.0000 | 0.0000 | 0 | partial |
| 5cm | light | 201 | 0.2714 | 1.0000 | 0.2714 | 478 | partial |
| 10cm | moderate | 101 | 0.7631 | 1.0000 | 0.7631 | 2758 | pass |
| 10cm | moderate | 102 | 0.7067 | 1.0000 | 0.7067 | 2554 | pass |
| 10cm | heavy | 202 | 0.6363 | 0.9754 | 0.6467 | 2396 | pass |
| 20cm | moderate | 203 | 0.6729 | 1.0000 | 0.6729 | 5340 | partial |
| 40cm | heavy | 204 | 0.7422 | 0.9981 | 0.7432 | 16125 | partial |

seed 42 不是唯一报告对象；10cm moderate 的两个额外 seed 仍产生可用但更低的 recall，说明基线存在 seed 敏感性，不能用单序列结果代表泛化性能。

## 10. Dry 误报与事件级限制

6 条 dry 序列（light/moderate/heavy，各含原始和额外 seed）最终 predicted water mask 全部为空，合计误报像素为 0，且 gate 全部为 partial，没有把无证据结果当作可信 dry。

但是事件分类层在这 6 条序列中累计产生 322 条 false water tracks。18 条有水序列的 water-event F1 平均仅 0.0863。最终 mask 未误报主要依赖稀疏证据强度阈值，而不是事件分类已经足够准确。因此当前分类器只适合作为可解释研究 baseline，不能进入正式链路。

## 11. 消融与静态材质捷径检查

在 18 条有水序列上，完整时域平均 IoU 为 0.7149；仅第一帧为 0；固定 seed 乱序为 0.4262；每帧颜色均值归一化后为 0.7096。完整时域明显优于单帧和乱序，证明帧顺序与动态变化是关键输入。颜色均值归一化没有明显破坏结果，且单帧结果为零，因此未发现仅靠静态水面材质获得高分的证据。

颜色归一化分支本身仍运行完整时域检测，结果接近 full 只说明算法对全局颜色均值不敏感，并不证明已经消除所有合成材质域偏差。

## 12. 合成域限制与下一步

Phase 2C-1 的雨丝、飞溅和涟漪是可重复的简化渲染，事件形态、曝光、噪声、风、雨幕、车灯和真实路面纹理均不充分。当前事件匹配精度低，light rain 覆盖不足，heavy rain 可能产生动态饱和；合成结果不能替代真实雨天视频验证。

下一步应先采集固定机位、含 dry/wet/shallow-water、light/moderate/heavy、白天/夜间的真实连续视频，并进行不泄漏的人工事件与水域标注；复核曝光校正、轨迹碎片合并、事件级 precision/recall 和 gate 校准。只有在经典特征在真实域中仍能稳定提供可分信息后，才考虑轻量学习模型。若进入学习路线，应以当前可解释特征和 gate 作为对照及安全约束，而不是直接替换整个链路。
