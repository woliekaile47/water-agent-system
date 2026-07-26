# Phase 2D-C-13：一键仿真传感器—Agent 标准链路

## 1. 目标

本阶段把已经分别验证过的模块串成一条可重复执行的离线命令：

1. 虚拟机读取 dry 场景的原始仿真 LiDAR rosbag；
2. 现有 S2 模块生成无水 Ground DEM；
3. 现有时序视觉模块从动态仿真 RGB 帧自动生成 SAM2 box、正点和负点；
4. WSL 使用 GPU 和 SAM2 视频传播生成连续 Camera water mask 候选；
5. 虚拟机使用现有岸线射线—DEM、水位估计和水深反演；
6. 候选质量门通过后，写入标准 S4 水深文件；
7. 现有 S5–S8、Agent 和 SQLite 审计链路继续执行。

这不是一条“比赛专用算法通道”。新增代码只负责跨 Windows、WSL 和虚拟机编排，感知、几何、水文、推理与预警计算继续复用项目原有实现。

## 2. 为什么由 Windows 总控

SAM2 GPU 环境位于 WSL，ROS2、rosbag 和完整项目位于 VMware Ubuntu。当前虚拟机不能直接连接 WSL 的 SSH 端口，而 Windows 可以同时连接两者。因此 Windows PowerShell 只承担文件转交和命令编排：

- 不进行模型推理；
- 不修改算法结果；
- 不读取 Ground Truth；
- 不调用真实设备；
- 不发出真实预警。

## 3. 固定验收样本

- case：`sim_water_20cm_001`
- rain：`moderate`
- seed：`303`
- RGB window：129–169
- anchor：149
- RGB：动态降雨仿真道路画面
- LiDAR：dry baseline 原始仿真 rosbag
- Ground Truth used for prediction：`false`

固定样本用于验证工程闭环，不表示只支持这一场景。后续可在同一配置契约下增加其他仿真样本。

## 4. 运行方式

在 Windows PowerShell 中执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_simulation_agent_e2e_windows.ps1
```

脚本自动生成唯一 run ID。也可以显式指定：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_simulation_agent_e2e_windows.ps1 `
  -RunId simulation_e2e_demo_001
```

运行前要求：

- VMware Ubuntu 已启动，SSH 可访问；
- WSL SSH 已启动并监听 2222；
- WSL SAM2 虚拟环境和 RTX 4060 CUDA 可用；
- 两个 SSH 私钥路径有效；
- 固定输入数据存在；
- 输出 run ID 尚未使用。

## 5. 输出

每次运行写入独立目录：

`outputs/phase2d_c13_one_click_runs/<run_id>/`

主要内容：

- `s2/`：由原始 dry LiDAR rosbag 生成的 Ground DEM；几何阶段使用同一次构建产生的有限值插值 DEM，原始稀疏 DEM 和有效性 mask 同时保留；
- `s3_prompt/`：时序视觉自动提示及诊断；
- `s3_video/`：SAM2 连续帧候选 mask；
- `s4_geometry/`：岸线、估计水位、DEM 水深、面积和体积；
- `runtime/`：标准 S4 接口、S5–S8、Agent、报告和 SQLite；
- `completion_summary.json`：一键闭环结果摘要；
- `logs/`：各标准阶段日志。

## 6. 安全与结果语义

本阶段全部结果均为：

- `data_domain = simulation`
- `authoritative = false`
- `eligible_for_real_warning = false`
- `warning_mode = simulation_record_only`
- `external_notification_allowed = false`
- `real_device_action_allowed = false`

Camera 看不到的独立盆地仍不会被自动补全。只有 Camera 可观测范围和全局场景状态满足现有候选门要求时，结果才允许进入本次“仿真记录链路”；它仍不等同于真实预警。

## 7. 验收标准

一键运行成功必须同时满足：

1. Ground DEM 确实由 dry 仿真 LiDAR rosbag重新生成；
2. 自动提示声明未使用 Ground Truth；
3. SAM2 返回完整 41 帧候选 mask；
4. ray–DEM 几何处理完成；
5. anchor frame 通过现有候选质量门且全局场景完整；
6. 标准 S4 水深接口文件生成；
7. S5–S8 和 Agent 状态为 success；
8. SQLite 审计数据库存在且非空；
9. warning 仅为 `simulation_record_only`；
10. 未启动真实 Camera、LiDAR、RTSP、Gazebo 或外部通知。

## 8. 后续路线

一键仿真闭环通过后，下一阶段才是“外部 API 沙箱接入”：先接天气、地图或通知 API 的测试环境，并保持真实预警关闭。再往后才是学校设备回归、真实浅水视频验证与多 Camera 覆盖。当前阶段不声称已经完成真实环境部署。

## 9. 固定样本一键验收

2026-07-26 使用 `c13_acceptance_20260726_02` 从一条 Windows 命令完成全部六步，结果如下：

- dry 仿真 LiDAR rosbag 成功重新生成 Ground DEM；
- 自动时序提示状态：`pass`；
- SAM2 GPU 连续传播：41 帧完整，耗时 10.28 秒；
- 相邻 mask IoU 中位数：0.980404；
- mask 面积变异系数：0.018539；
- estimated water level：-0.240260 m；
- mean / max depth：9.3993 cm / 20.0126 cm；
- S5 area / volume：6.89 m² / 0.666800 m³；
- Camera reprojection IoU：0.942389；
- outer boundary P95：3.605551 px，仅作诊断，不触发旧 3 px 否决；
- camera-visible status：`pass`；
- global-scene status：`complete`；
- Agent：`success`；
- SQLite：存在且非空；
- warning：`none`，模式为 `simulation_record_only`；
- Ground Truth used：`false`；
- 真实设备、真实 API 和外部通知：均未启动。

SAM2 运行时提示可选 C 扩展 `_C` 未加载，因此跳过内部小孔洞后处理；该提示未阻断本次传播和标准链路，但应保留在环境已知限制中，不能静默当作完整扩展安装。
