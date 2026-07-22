# Phase 2D-C-9A：统一水状态接口与双门控 Shadow Mode

## 目标

在不修改现有 S5-S8、Agent 和 Dashboard 的前提下，将 Phase 2D-C-8 已确认的候选门控结果转换为稳定的 `canonical_water_state_v1` 接口，并与旧运行时 gate 并行记录。

本阶段只做 shadow integration：新候选门控可以被观察和审计，但不能控制正式预警链路。

## 核心语义

- `observation_scope = camera_observable_region`：明确结果来自 Camera 可见范围。
- `global_estimate_status = complete / partial / unavailable`：区分完整、部分和不可用结果。
- `result_semantics = global_scene_estimate`：仅在候选门控通过且不存在不可观测盆地时使用。
- `result_semantics = camera_visible_estimate`：Camera 可见主水域有效，但全局场景不完整。
- `area_volume_semantics = observable_lower_bound`：不可观测盆地存在时，面积和体积只能作为可见范围下界。
- `measurement_status = rejected_candidate`：几何数值可以保留用于诊断，但不能成为正式测量。

## 双门控 Shadow 设计

每帧同时保存：

1. 旧运行时 gate 的状态和拒绝原因；
2. Phase 2D-C-8 候选 gate 的可见范围状态、全局状态、拒绝原因和 warning；
3. 两者是否发生状态分歧；
4. prediction 输入与配置文件的 SHA-256 provenance。

无论新旧 gate 是否通过，C9-A 都强制：

```text
authoritative = false
eligible_for_downstream = false
downstream_block_reason = phase2d_c9a_shadow_mode_not_authoritative
```

因此 C9-A 不会改变现有预警行为，也不会把 seed 303 仿真结果直接接入 S5-S8。

## 安全约束

- Ground Truth 字段不得进入 canonical prediction state。
- `deployment_mode` 只能为 `shadow`。
- shadow mode 禁止启用 downstream。
- `global_estimate_status=partial` 时必须使用 `camera_visible_estimate` 与 `observable_lower_bound`。
- 不可用或被拒绝结果不得暴露为正式结果语义。
- 当前仍为 synthetic-domain、not-real-world-validated、non-authoritative。

## 后续

C9-B 将在不改变预警结果的情况下，让 S5-S8 读取 shadow canonical state 并生成对照审计；只有完成接口回归、故障注入和人工验收后，才讨论正式 gate 迁移。
