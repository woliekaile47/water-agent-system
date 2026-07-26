# Phase 2D-C-15：多场景系统验收

## 目的

使用同一套离线正式流程处理 5、10、20、40 cm 四个固定仿真场景，验证系统不但能产生结果，也能在输入不可信或全局不可观测时安全停止。

## 冻结输入

四个场景均使用 moderate rain、seed 303、frame 129–169，锚点为 frame 149。选样和预期状态在运行前冻结，不读取 Ground Truth 调参。

## 预期行为

- 5 cm：浅水视觉边界不足，Camera 可见结果应 reject，Agent 不运行。
- 10 cm：Camera 可见且全局完整，应进入标准 S4–S8 与 Agent。
- 20 cm：Camera 可见且全局完整，应进入标准 S4–S8 与 Agent。
- 40 cm：Camera 可见部分有效，但存在不可观测第二盆地；全局应为 partial，Agent 不运行。

5 cm 和 40 cm 的停止属于质量门正确工作，不是程序崩溃。系统不得为了“全流程都变绿”而绕过质量门。

## 实际链路

无水仿真 LiDAR rosbag → Ground DEM → 动态仿真 RGB → 自动时序提示 → WSL GPU SAM2 视频传播 → Camera mask → 岸线 ray–DEM → 水位、深度、面积和体积 → 候选质量门 → 符合条件时进入标准 S5–S8 与 Agent。

## 安全边界

- Ground Truth 不进入 prediction。
- 所有结果均为 simulation-only、non-authoritative。
- 不启动真实 Camera、LiDAR、RTSP、ROS 或 Gazebo。
- 不触发外部通知、真实设备动作或正式预警。
- 40 cm 不自动填充 Camera 看不到的第二盆地。

## 验收

运行后自动生成 JSON 与 Markdown 汇总。四个场景都与预先冻结的安全行为一致时，矩阵状态才为 `pass`。

## 实际结果

验收 ID：`c15_acceptance_20260726_01`。矩阵结果为 `pass`，4/4 场景符合冻结预期。

| 水深 | Camera 状态 | 全局状态 | Agent | 最大水深 | Camera 重投影 IoU | 结果 |
|---|---|---|---|---:|---:|---|
| 5 cm | reject | unavailable | 质量门阻断 | 5.83 cm | 0.8763 | 浅水视觉自一致性不足，安全拒绝 |
| 10 cm | pass | complete | success | 10.49 cm | 0.9115 | 标准 S5–S8、Agent、SQLite 完成 |
| 20 cm | pass | complete | success | 20.01 cm | 0.9424 | 标准 S5–S8、Agent、SQLite 完成 |
| 40 cm | pass | partial | 质量门阻断 | 39.08 cm | 0.9226 | Camera 可见主水域有效；全局存在不可观测/歧义盆地 |

40 cm 的 outer-boundary P95 为 10.03 px，只产生辅助警告；系统没有恢复旧 3 px 单指标否决。真正阻断全局链路的是 Camera 覆盖范围外的候选盆地与多盆地歧义。

SAM2 运行期间仍报告可选 C 扩展不可用并跳过后处理；视频传播本身成功完成。该环境提示应保留记录，不能把它描述为完整可选后处理已启用。
