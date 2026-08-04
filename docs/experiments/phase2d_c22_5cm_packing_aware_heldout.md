# Phase 2D-C-22：5 cm 安全正点组合与 seed311 held-out 验证

## 目的

C21 的 seed310 已找到跨 40 秒重复出现的浅水核心，但旧贪心算法先选择中心点，
导致剩余候选无法凑齐 3 个相距至少 10 px 的安全正点。本阶段只修复正点组合策略，
不降低任何安全阈值；随后在全新 seed311 上只运行一次完整 held-out 链路。

本阶段始终保持：

- prediction 不读取 Ground Truth；
- 不人工补点或修改提示；
- 不修改 3 px 边界距离、10 px 点间距、0.50 概率或 2/3 跨窗口支持率；
- 不根据 seed311 评价结果调参或重跑；
- GT 只在 SAM2 和几何 prediction 全部冻结后由独立 evaluation 读取；
- 所有结果均为研究候选，`authoritative=false`、`eligible_for_downstream=false`。

## C22 修改

C22使用独立Prompt配置：

`configs/temporal_sam2_prompt_c22_packing_aware.yaml`

旧C17/C21继续使用`configs/temporal_sam2_prompt_corroborated.yaml`。两个配置除
`algorithm_version`和C22新增的`positive_point_selection_method`外，所有安全阈值
完全相同。

正点选择保持旧行为优先：旧贪心已经得到至少 3 个点时，坐标和输出不变。只有旧
贪心少于 3 点时，才启用确定性、质量有序的可行集合搜索：

1. 只在已经满足边界距离、概率和跨窗口支持率的候选中搜索；
2. 先寻找 3 个两两间距至少 10 px 的点；
3. 找到安全三点后，再用 maximin 尝试扩展到目标 5 点；
4. 若确实不存在合法三点集合，保留诊断点并安全拒绝；
5. 不放宽任何阈值。

seed310 只用于冻结数组上的 prompt-only 开发验证，没有重跑 60 秒时序预测，也没有
继续运行 SAM2 或几何。结果从 2 个安全正点变为 3 个，prompt 由 `reject` 变为
`pass`，证明阻断来自贪心组合局限，而不是候选数据不足。

### 配置快照与冻结哈希

seed310 prompt-only 和 seed311 首次执行时，C22配置尚使用共享文件名。收尾阶段将
相同字节内容移动到上述C22独立配置，恢复旧共享配置。冻结manifest记录的是配置内容
SHA-256而不是文件路径；C22独立快照SHA-256仍为：

`d90c94266677c1baf7cdb8cd9005f606b316827bf9833d0f0c8fbae5b576cc39`

它与seed311 `fusion_manifest.json`中的`input_config_sha256.prompt`完全一致，因此
配置隔离没有修改、重算或伪造任何冻结实验输出。

未来运行C22/C23时必须显式使用：

```bash
--prompt-config configs/temporal_sam2_prompt_c22_packing_aware.yaml
```

## Seed311 冻结协议

- case：`sim_water_5cm_001`
- rain level：`moderate`
- dynamic seed：`311`
- 视频：60 秒、20 FPS、1,201 帧
- 13 个固定候选窗口，每窗 41 帧
- 代表性视频传播窗口：41 帧
- 角色：`held_out_validation`
- 数据生成、融合、SAM2、几何 prediction 均只运行一次

## Prediction-side 结果

### 多窗口融合与自动提示

| 指标 | 结果 |
|---|---:|
| 候选窗口 | 13 |
| 可参与融合窗口 | 9 |
| 最终核心真实支持窗口 | 8 |
| 支持时间跨度 | 58 s |
| 融合核心面积 | 563 px |
| 歧义核心数 | 0 |
| 自动安全正点 | 3 |
| 自动负点 | 8 |
| prompt 状态 | `pass` |
| GT / 人工提示 | false / false |

seed311 的旧贪心本身已成功产生 3 个点，因此没有触发 C22 fallback。这说明 C22 对
旧成功路径没有造成回归；fallback 的有效性由冻结 seed310 prompt-only 验证和单元
测试提供证据。

### SAM2 视频传播

| 指标 | 结果 |
|---|---:|
| 帧数 | 41 |
| 相邻 mask IoU 中位数 | 0.9390 |
| 相邻 mask IoU 最小值 | 0.6669 |
| mask 面积 CV | 0.1220 |
| 推理耗时 | 7.21 s |
| GPU 峰值 allocated | 1,148.94 MiB |
| 完成状态 | true |

SAM2 可选 C 扩展不可用时跳过了可选 hole postprocess；模型推理和 41 帧传播正常完成，
没有自动切换到 CPU，也没有发生 CUDA OOM。

### DEM 几何与时序稳定性

| 指标 | 结果 |
|---|---:|
| 可计算帧 | 41 / 41 |
| 水位中位数 | -0.394006 m |
| 水位标准差 | 0.1347 cm |
| 相邻水位变化 P95 | 0.2249 cm |
| 面积中位数 / CV | 1.57 m² / 0.0200 |
| 体积中位数 / CV | 0.039372 m³ / 0.0538 |
| 最大水深中位数 | 4.989 cm |
| 最大水深范围 | 4.667–5.223 cm |
| 射线求交成功率 | 41 帧均为 1.0 |
| candidate / selected basin | 每帧 1 / 1 |
| 不可观测或歧义 basin | 0 |

水位标准差、相邻变化、面积 CV 和体积 CV 均满足既有冻结稳定性参考值。没有 NaN、
Inf、负水深或失败帧。

prediction-side 质量门仍为 41/41 `reject`，唯一硬拒绝原因是：

`camera_reprojection_iou_below_threshold`

Camera 重投影 IoU 最小/中位/最大为 0.5732/0.7321/0.8865，均低于现有 0.90
候选阈值。外岸线 P95 继续作为辅助诊断，不是本批唯一拒绝的直接原因。

## 独立 Ground Truth 评价

所有 prediction 文件及 SHA-256 均在首次读取 GT 前验证并冻结。评价没有重算
prediction、没有修改 prompt 或质量门。

### Camera mask

| 指标 | 41 帧中位数 | 均值 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| IoU | 0.7310 | 0.7403 | 0.5789 | 0.8845 |
| precision | 0.9494 | 0.9491 | 0.9217 | 0.9704 |
| recall | 0.7683 | 0.7707 | 0.5940 | 0.9222 |
| F1 | 0.8446 | 0.8475 | 0.7333 | 0.9387 |
| outer boundary P95 | 16.05 px | 15.62 px | 5.00 px | 26.07 px |

高 precision、较低 recall 表明主要问题是 SAM2 对 5 cm 浅水范围的欠分割，而不是
大面积普通路面误覆盖。41 帧均未满足冻结的离线视觉研究标准。

### 水位、面积、体积和水深

GT 水位为 `-0.3938947141 m`。水位绝对误差：

| 指标 | 结果 |
|---|---:|
| 最小 | 0.00054 cm |
| 中位数 | 0.08891 cm |
| 均值 | 0.10536 cm |
| 最大 | 0.33269 cm |
| 3 cm 内帧数 | 41 / 41 |

| 指标 | 中位相对误差 | 均值 | 最大值 |
|---|---:|---:|---:|
| 面积 | 0.63% | 1.39% | 5.70% |
| 体积 | 3.53% | 4.18% | 12.96% |

可见水域平均水深绝对误差中位数为 0.0730 cm；最大水深绝对误差中位数为
0.0889 cm。C7 冻结输出只为 anchor 保存逐栅格数组，因此本阶段没有为评价重算其余
40 帧的逐栅格 DEM mask/depth 指标。

## 结论

seed311 是一次有效且未重跑的 held-out 结果：

- 自动多窗口证据、自动提示、SAM2 视频传播和 DEM 几何全链路已跑通；
- 41/41 帧水位误差小于 3 cm，且水位、面积、体积时序稳定；
- 但 5 cm Camera mask 仍存在系统性欠分割，Camera/重投影一致性不足；
- 现有质量门据此保持 41/41 reject，未把高精度水位评价反向变成正式 measurement；
- 当前可表述为“5 cm 水位精度目标在该 held-out 样本上达到，但自动视觉放行目标未达到”；
- 不能表述为项目已经完成无人值守的 5 cm 正式部署。

下一阶段应分析跨多个 2 秒窗口持续出现的浅水核心如何形成更完整、稳定的 SAM2
提示范围，重点提高 recall，不应降低质量门或按 seed311 调参。

## 主结果 SHA-256

| 文件 | SHA-256 |
|---|---|
| seed311 sequence manifest | `d4a6df3593e9810f0ecacf591caa35b03740037fc352ebf199fde5f842d86183` |
| multi-window fusion | `e9cb12e6e5a4834a79e7438a6b910bef89487aa9f4abd037c1b30ca2dbee8091` |
| automatic prompt | `4b5b9639bbd152ae4cd9d44f21b1595531f26144832db6440a4d180e0a936d41` |
| SAM2 video summary | `6e965b74d62199bde551b1d2623f9b977dfce610198fbf86939b593d6d64c28e` |
| geometry summary | `12caf0de265b597996fc40bc022504ad09273f5854ff15af0c637ed26f9b90e0` |
| Camera GT evaluation | `c989c3ed8aed23ca4b367498767c94a6d80f8e4b530ded7d7297877355cd14b6` |
| geometry GT evaluation | `2c44469d686f27b49ea47d76697f51307b2d7fb701cba2195c9d3c2046fa5cab` |

## 安全边界

- 没有启动 ROS、Gazebo、Camera、LiDAR 或 RTSP；
- 没有接入 S5-S8、Agent、数据库或 Dashboard；
- 没有修改质量门阈值；
- 没有执行 commit 或 push。
