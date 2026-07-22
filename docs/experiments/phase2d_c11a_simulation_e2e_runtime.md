# Phase 2D-C-11A：仿真端到端运行域隔离

## 目标

本阶段为后续“仿真数据跑通 Agent 全流程”建立独立运行通道。它只做资格判断和安全路由，不执行正式 S5–S8，不发送预警，也不控制真实设备。

## 为什么需要这一层

C9 的 shadow 模式故意把所有结果锁定为 `authoritative=false`、`eligible_for_downstream=false`。这保证了研究结果不会误入正式预警链，但也意味着质量门控通过的仿真结果无法进入后续仿真验证。C11-A 保留原锁定字段不变，新增一套仅属于 `simulation_e2e` 运行域的资格字段。

## 三类路由语义

- `complete`：可进入仿真 S5/S6，也可进入需要全局水域语义的仿真 S7/S8 评价。
- `partial`：仅允许使用 Camera 可见范围内的水深、面积和体积下界；禁止冒充全场景结果，不能进入全局 S7/S8 路由。
- `unavailable/reject`：完全阻断，并清空路由包中的测量值。

## 永久安全边界

所有 C11-A 输出均固定满足：

- `data_domain=simulation`
- `authoritative=false`
- `eligible_for_downstream=false`
- `eligible_for_real_warning=false`
- `warning_action_mode=simulation_record_only`
- `external_notification_allowed=false`
- `real_device_action_allowed=false`
- `formal_output_writes_allowed=false`

配置试图切换为 `production`、真实数据域、真实通知、真实设备动作或正式输出写入时，构建过程会直接失败。

## 输入与 Ground Truth 隔离

输入仅为已经冻结的 C9 canonical prediction records。构建器不读取评价目录、Camera/DEM Ground Truth、水位真值、面积真值或体积真值；带有 GT 使用标记或非 synthetic provenance 的记录会被拒绝。

## 本阶段输出

输出目录为 `outputs/phase2d_c11_simulation_e2e_runtime_*`，属于可重复生成的本地工件：

- `simulation_runtime_envelopes.json/jsonl`
- `simulation_runtime_current_by_sample.json`
- `simulation_runtime_summary.json`

构建前后会核对现有五个正式 S5–S8 JSON 的 SHA-256，任何变化都会导致失败。

## 不在本阶段实现

- 不执行 S5 面积体积正式写入；
- 不接入真实天气 API；
- 不执行分钟级 S7 趋势预测；
- 不生成或发送真实预警；
- 不修改 Agent、数据库、Dashboard；
- 不启动 ROS、Gazebo、Camera、LiDAR 或 RTSP。

下一阶段将在该隔离契约之上构造足够长的仿真时间线，再逐步运行仿真版 S5–S8。
