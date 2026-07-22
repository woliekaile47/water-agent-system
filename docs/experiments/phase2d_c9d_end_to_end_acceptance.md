# Phase 2D-C-9D：端到端故障测试与比赛演示验收

## 范围

本阶段验证 C9-A canonical state、C9-B S5-S8 shadow、C9-C sidecar monitor/DB/API/Dashboard 能否形成安全的离线演示闭环。

验收范围是 `synthetic_shadow_demo`，不是生产部署或真实道路验证。

## 正常链路验收

自动检查：

- 492 条 canonical record 和 12 个样本完整；
- candidate pass/reject 与 global partial 计数符合冻结结果；
- 所有 canonical record 均为 prediction-only、non-authoritative、downstream blocked；
- 12 个 S8 shadow 状态全部为 `warning_suppressed`；
- Agent sidecar monitor 为 healthy；
- 正式 S5-S8 JSON 与正式 Agent SQLite 未改变；
- HTTP、Dashboard、正式 Agent 均未启动；
- rejected API sample 不暴露测量值；
- 独立 SQLite 所有样本均保持 warning suppressed 和 downstream blocked。

## 故障注入

所有故障仅在临时目录或内存副本中注入：

1. 缺失必需 JSON；
2. JSON 损坏；
3. GT provenance 泄漏；
4. shadow state 被错误标记为 downstream eligible；
5. partial 结果冒充 global complete；
6. shadow 中启用 warning generation；
7. API 暴露 downstream-enabled state；
8. SQLite 写入 authoritative shadow record；
9. 正式文件哈希不一致。

只有全部故障都被拒绝，C9-D 才能通过。

## 演示结论边界

通过自动验收后允许声明：

> 系统已具备仿真环境下的两阶段感知、SAM2 视频分割、DEM 水深反演、候选质量门控、统一状态接口以及安全 shadow 展示闭环。

仍然禁止声明：

- 已完成真实道路自动积水识别；
- 已完成生产级预警部署；
- 已批准新 gate 生成正式预警；
- 仿真指标可以代替真实场景指标。

## 人工演示

自动验收不启动 GUI。需要人工展示时，在 Ubuntu 图形桌面运行：

```bash
cd /home/wlkl/water_agent_ws/water_agent_system
streamlit run dashboard/app.py
```

进入“C9 Shadow：统一状态监控”页面，确认 monitor healthy、warning generated false、formal DB unchanged true。

## 自动验收结果

- Acceptance status：`pass`。
- 端到端不变量：22/22 通过。
- 故障注入：9/9 被正确拒绝。
- 受保护的 5 个正式 S5-S8 JSON 和正式 Agent SQLite：运行前后哈希全部一致。
- Competition demo readiness：`ready_for_synthetic_shadow_demo`。
- Production readiness：false。
- Real-world validated：false。
- Formal warning activation allowed：false。
- HTTP server / Dashboard / formal Agent：均未启动。
