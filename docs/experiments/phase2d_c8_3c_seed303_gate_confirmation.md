# Phase 2D-C-8-3C：seed 303 候选质量门控独立确认

## 目的与协议

本阶段使用未参与候选门控阈值选择的 seed 303 最终确认集，验证 Phase 2D-C-8 候选门控能否在不读取 Ground Truth 的条件下筛出满足项目水位误差 3 cm 目标的 Camera 可见范围结果。

- 数据矩阵：4 个水深 × 3 个雨强 × 1 个新 seed。
- 每个序列固定 anchor frame 149，评价窗口为 frame 129–169，共 41 帧。
- 合计 12 个序列、492 帧。
- 自动提示、SAM2 视频传播、几何 prediction 与候选门控均在首次读取 GT 前冻结。
- 候选门控配置 SHA-256：`75c92e7f4f21ef9c05a65d564a78a4634dc6b3c3c5382fb31b6b1a2f8c45c728`。
- 本阶段没有重新运行 SAM2，没有根据 seed 303 GT 修改提示、prediction、阈值或 3 px 旧门控。

## Prediction-side 冻结结果

| 状态 | 帧数 |
|---|---:|
| camera-visible pass | 361 |
| camera-visible reject | 131 |
| global complete | 238 |
| global partial | 123 |
| global unavailable | 131 |

131 个 reject 全部包含 `camera_reprojection_iou_below_candidate_threshold`。55 帧的外岸线重投影 P95 高于 5 px，但该指标只产生 advisory warning，单独造成 reject 的帧数为 0。40 cm 的 123 帧均保留 Camera 可见范围 pass，同时由于存在 Camera 不可观测候选盆地而标记为 global partial。

## 独立 GT 确认

| 水深 | 可见 pass / 总帧 | pass 中 3 cm 内 | pass 水位误差中位数 / 最大值（cm） | pass Camera IoU 中位数 | pass 可见面积相对误差中位数 | pass 可见体积相对误差中位数 |
|---|---:|---:|---:|---:|---:|---:|
| 5 cm | 9 / 123 | 9 / 9 | 0.446 / 0.598 | 0.879 | 9.49% | 18.56% |
| 10 cm | 106 / 123 | 106 / 106 | 0.296 / 0.805 | 0.926 | 2.46% | 6.01% |
| 20 cm | 123 / 123 | 123 / 123 | 0.360 / 1.163 | 0.953 | 1.84% | 3.75% |
| 40 cm | 123 / 123 | 123 / 123 | 0.437 / 0.863 | 0.968 | 1.95% | 2.59% |
| **总计** | **361 / 492** | **361 / 361** | **0.375 / 1.163** | **0.951** | **2.16%** | **3.50%** |

全部 492 帧的水位绝对误差中位数为 0.438 cm，最大值为 1.764 cm；候选门控放行的 361 帧中没有任何一帧超过 3 cm。独立确认结论为 `pass`。

## 浅水与可观测性解释

5 cm 仍是主要限制：light 仅 3/41 pass，moderate 仅 6/41 pass，heavy 0/41 pass。其 Camera GT IoU 整体中位数为 0.841，明显低于 10/20/40 cm。候选门控没有将浅水语义不可靠帧放行；即使 5 cm 的几何水位误差在本仿真数据中低于 3 cm，也不能据此宣称自动浅水分割已经解决。

40 cm 的 Camera 可见主盆地结果准确，但第二盆地不可观测。因此面积和体积只能解释为 Camera 可见范围估计，不能冒充全场景完整测量；`global partial` 语义必须继续保留。

## 结论与边界

1. 证据支持：3 px 旧 boundary gate 对“水位误差不超过 3 cm”的项目目标过于保守，像素边界距离不能直接等价为厘米水位误差。
2. 冻结的 C8 候选门控在 seed 303 确认集上没有出现可见范围 false pass，且保持了对 5 cm 的保守拒绝。
3. 本结论只确认离线候选门控设计，结果仍为非 authoritative，`eligible_for_downstream=false`。
4. 在 Phase 2D-C-9 中应先以 shadow mode 接入统一结果接口，保留旧门控和新候选门控并行审计；完成契约、回归和故障注入前不得直接进入正式 S5-S8 预警。
