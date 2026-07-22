# Phase 2D-C-9B：S5-S8 Shadow 对照集成

## 目标

验证 `canonical_water_state_v1` 能提供现有 S5/S7/S8 所需的核心字段，同时保证新两阶段感知结果不会覆盖旧 MVP 输出或生成正式预警。

## 实现边界

- S5：只检查 `mean_depth_cm`、`max_depth_cm`、`water_area_m2` 和 `water_volume_m3` 的字段兼容性。
- S6：不注入现有 offline mock weather，也不执行天气修正。
- S7：只检查连续水深历史长度，不执行预测。
- S8：固定输出 `warning_suppressed`，不产生预警等级、行动建议或正式 warning JSON。
- Agent、数据库和 Dashboard 本阶段不修改。

## 为什么不执行 S7 预测

seed 303 的固定窗口包含 41 帧，帧率 20 FPS，对应约 2 秒。现有 S7 需要 1、5、10 分钟历史水深。将 2 秒数据伪装成 10 分钟历史会制造没有依据的上涨斜率和预警，因此 C9-B 必须返回：

```text
history_ready = false
forecast_executed = false
forecast_results = []
```

## 语义保护

- `global_scene_estimate` 可映射到 S5 complete-estimate 字段，但仍不能进入 S6-S8。
- `camera_visible_estimate` 的面积和体积保持 `observable_lower_bound`，不能冒充全局值。
- rejected canonical state 不向 S5 暴露面积、体积和水深数值。
- 所有 S8 shadow 结果均为 `warning_suppressed`。

## 文件保护

运行前后对以下正式 MVP 输出计算 SHA-256：

- `outputs/json/water_area_volume_result.json`
- `outputs/json/weather_correction_result.json`
- `outputs/json/deterministic_forecast_result.json`
- `outputs/json/final_forecast_result.json`
- `outputs/json/warning_decision_result.json`

任何哈希变化都会使 shadow 运行失败。C9-B 只写入独立、被 Git 忽略的实验输出目录。

## seed 303 实际结果

- 12 个样本的最新 canonical state 均完成 shadow 检查。
- S5 字段兼容候选 7 个，门控阻断 5 个。
- S7 `not_ready` 12 个，没有生成任何预测曲线。
- S8 `warning_suppressed` 12 个，允许生成 warning 的数量为 0。
- downstream eligible 数量为 0。
- 5 个既有正式 S5-S8 JSON 的运行前后 SHA-256 完全一致。

“S5 字段兼容”只表示字段形状可被后续模块理解，不表示结果已经被批准进入正式链路。

## 下一步

C9-C 将把 canonical/shadow 状态加入 Agent 审计、数据库/API 和 Dashboard 的只读展示，仍不改变正式预警决策。
