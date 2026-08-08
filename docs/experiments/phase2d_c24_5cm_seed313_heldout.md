# Phase 2D-C24：5 cm seed313 正式 held-out 验证

## 目的与冻结协议

本阶段在 C23 参数、代码和模型全部冻结后，使用全新 seed313 验证 5 cm 浅水自动链路。测试覆盖 light、moderate、heavy 三种雨强，每组 60 秒、20 FPS、1,201 帧。所有窗口由固定规则选择，不看 RGB 主观挑选，不人工补点，不按雨强调参；prediction 完成并记录哈希后，GT 才由独立 evaluation 读取。

冻结代码 commit 为 `c7319b2`。SAM 2 checkpoint、Ground DEM、相机标定、mask 稳定化和 quality gate 均未改变。本阶段结果继续保持 `authoritative=false`、`eligible_for_downstream=false`。

## Prediction-side 结果

| 雨强 | 融合窗口 | 真实支持窗口 | 时间跨度 | 安全正点 | Prompt | SAM2 | Geometry gate |
|---|---:|---:|---:|---:|---|---|---|
| light | 6 | 4 | 53 s | 1 | reject | 未运行 | 未运行 |
| moderate | 10 | 10 | 48 s | 3 | pass | 41 帧 | 17 pass / 24 reject |
| heavy | 4 | 4 | 58 s | 3 | pass | 41 帧 | 3 pass / 38 reject |

小雨融合找到了跨 53 秒重复出现的浅水核心，但只有 1 个点同时满足既有概率、边界距离、跨窗口支持和点间距要求，因此系统安全停止。该样本没有重跑、人工补点、降低最少 3 正点要求或替换 seed。

中雨和大雨的唯一 gate 拒绝原因为 `camera_reprojection_iou_below_threshold`。quality gate 阈值没有修改。

## 独立 GT 评价

| 指标 | moderate | heavy |
|---|---:|---:|
| Camera IoU 中位数 | 0.8924 | 0.8381 |
| Camera precision 中位数 | 0.9334 | 0.8894 |
| Camera recall 中位数 | 0.9500 | 0.9551 |
| Outer boundary P95 中位数 | 3.21 px | 5.00 px |
| 水位绝对误差中位数 | 0.3658 cm | 0.5032 cm |
| 水位绝对误差最大值 | 0.6942 cm | 1.2826 cm |
| 水位标准差 | 0.1550 cm | 0.2362 cm |
| 相邻水位变化 P95 | 0.1220 cm | 0.2841 cm |
| 可见面积误差中位数 | 1.90% | 0.63% |
| 可见体积误差中位数 | 6.66% | 4.41% |
| 几何面积 CV | 0.0276 | 0.0461 |
| 几何体积 CV | 0.0648 | 0.0976 |

中雨和大雨共 82 帧，82/82 水位绝对误差小于 3 cm，且水位、面积、体积稳定性满足冻结目标。Camera 独立评价的总体 IoU 中位数为 0.8741，但只有 15/82 帧满足现有 Camera 组合研究标准。

## 正式结论

`overall_status = partially_accurate_not_accepted`。

- 水位精度目标：对可评价的中雨和大雨通过；
- 水位、面积和体积稳定性：通过；
- 全雨强自动提示：失败，小雨没有足够安全正点；
- quality gate 稳定通过：失败，中雨和大雨共有 20/82 帧 pass；
- 正式 downstream 接入：禁止。

因此 seed313 是有效的正式 held-out 部分通过结果，但不能宣称 5 cm 全雨强无人值守链路已经成功。后续不得把 seed313 重新作为调参开发数据反复运行；应先冻结本结果，再设计新的通用改进，并用另一个预先声明的新 seed 验证。

## 冻结摘要 SHA-256

| 摘要 | SHA-256 |
|---|---|
| Camera GT evaluation | `b26e4a82c7f5d22bf3f6d335ba9be6558931f183c31edda63c1b947cfb3983b9` |
| Geometry prediction | `17fc9a1eb4b0af58c5bfc6900cc62c04189936c6e2b0f48d5f0f8aae77c56f63` |
| Geometry GT evaluation | `ccf86bd15f3826f96214fbbbad74f6fa8b3c4376caabe32ba1c90b0b4518f2d8` |
