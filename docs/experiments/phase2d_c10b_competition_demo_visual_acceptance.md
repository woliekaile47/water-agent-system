# Phase 2D-C-10B：仿真道路比赛演示人工显示验收

## 验收范围

2026-07-22 在 Ubuntu 图形桌面使用比赛专用入口运行：

```bash
bash scripts/run_phase2d_c10_competition_demo.sh
```

人工检查固定 Seed 303、frame 149 的 5/10/20/40 cm moderate rain 场景。本验收只检查页面加载、素材来源和结果语义，不重新运行 SAM 2、几何 prediction 或 Ground Truth evaluation。

## 显示结果

四个场景均正常显示：

- 仿真道路与积水 Camera 画面；
- 冻结的 SAM 2 视频传播候选 mask；
- DEM 水域重投影到 Camera 的 mask；
- 连续 41 帧水位、最大水深、Camera 可见面积和体积曲线。

页面未显示宿舍纸箱、`water_test`、人工提示盲测或真实设备素材。比赛专用入口没有历史宿舍对照页面导航。

## 门控语义

| 仿真场景 | Camera 可见状态 | 全局状态 | 展示结论 |
|---|---|---|---|
| 5 cm / moderate | reject | unavailable | 浅水候选自一致性不足，如实拒绝 |
| 10 cm / moderate | pass | complete | Camera 可见与全局估计完整 |
| 20 cm / moderate | pass | complete | Camera 可见与全局估计完整 |
| 40 cm / moderate | pass | partial | 主水域可见，但存在 1 个不可观测候选盆地 |

40 cm SAM 2 mask 中仍可见少量局部碎片和内部结构差异；本次展示不做形态学美化或人工修补，保留冻结预测的真实状态。`partial` 不能解释为完整道路全域面积。

## 显示语义修正

人工验收发现旧 `status_badge` 会把 `unavailable` 显示为绿色。C10B 仅修正 Dashboard 颜色映射：

- `reject`、`unavailable`、`blocked` 显示为红色；
- `partial`、`not_ready`、`warning_suppressed` 显示为黄色；
- `pass`、`complete`、`healthy` 显示为绿色。

该修改不改变 prediction、候选 gate、阈值、canonical state 或 S5-S8 状态。

## 结论

比赛专用仿真道路页面通过人工显示验收。当前版本适合合成数据旁路演示，继续保持：

- `ground_truth_used=false`；
- `authoritative=false`；
- `eligible_for_downstream=false`；
- 不生成正式预警；
- 不代表真实道路部署验证完成。
