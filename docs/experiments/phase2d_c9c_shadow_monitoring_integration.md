# Phase 2D-C-9C：Agent、数据库、API 与 Dashboard 旁路监控

## 目标

将 C9-A/B 的 canonical/shadow 结果提供给工程展示和审计模块，同时不修改正式 Agent 调度、不写现有审计数据库、不启动 HTTP 服务、不生成预警。

## 架构

```text
canonical shadow state
        +
S5-S8 shadow envelope
        |
        v
Agent sidecar monitor
        |
        +--> independent shadow SQLite audit
        +--> framework-neutral read-only API payload
        +--> Dashboard read-only page
```

## Agent sidecar

sidecar 只读取 C9-A/B 汇总并检查：

- authoritative 和 downstream eligible 数量均为 0；
- warning generation allowed 数量为 0；
- 正式 S5-S8 输出哈希未改变；
- 正式 S5-S8 没有被执行。

它不调用 `run_agent()`，也不改变 `configs/agent_config.yaml` 的阶段顺序。

## 数据库

审计数据写入独立的 `canonical_shadow_audit.db`，只包含 shadow run 和 sample state。现有 `data/db/water_agent_audit.db` 在运行前后计算 SHA-256，任何变化都会使任务失败。

## API 数据层

当前环境没有 FastAPI 依赖，本阶段不安装依赖、不启动服务。系统生成 `canonical_shadow_read_api_v1` JSON payload，作为后续 HTTP API 的稳定只读数据契约：

- rejected sample 不暴露测量值；
- warning level 和 action 均为空；
- authoritative 与 eligible_for_downstream 固定为 false。

## Dashboard

新增“C9 Shadow：统一状态监控”页面，展示：

- sidecar monitor 健康状态；
- canonical record/sample 数量；
- 新旧 gate 对照矩阵；
- S5/S7/S8 shadow 状态；
- 正式审计数据库是否保持不变；
- 只读 API payload。

页面仅读取生成的 snapshot；本阶段不自动启动 Streamlit。

## 安全边界

- 不修改正式 Agent pipeline。
- 不修改正式 audit SQLite。
- 不启动 API 或 Dashboard 服务。
- 不生成正式预警。
- 不接入真实设备。
- 所有结果仍是 synthetic、shadow、non-authoritative。

## seed 303 实际结果

- Agent sidecar monitor：`healthy`。
- canonical records：492；samples：12。
- 独立 shadow SQLite：12 条 sample audit 记录。
- API schema：`canonical_shadow_read_api_v1`。
- 正式 `water_agent_audit.db` 的运行前后 SHA-256 均为
  `e72f44708da100fba3c76175235d2941c1d5e17efec3802fa3e77e11a1e01124`。
- HTTP server started：false。
- formal Agent executed：false。
- formal warning generated：false。
- authoritative / downstream eligible：false / false。
