# Phase 2D-C-11B：仿真传感器数据直通现有流水线

## 1. 目的

本阶段验证的不是一套“仿真专用业务通道”，而是把仿真产生的原始传感器数据放入现有项目接口，依次执行：

dry LiDAR rosbag → Ground DEM → 动态 RGB → 自动时序提示 → SAM 2 视频传播 → 岸线与 ray–DEM 水位反演 → 标准 S4 深度文件 → 现有 S5–S8 → Agent → SQLite 审计。

仿真运行目录与正式目录隔离，仅用于防止测试数据覆盖正式数据；S5–S8 的计算模块保持复用。

## 2. 实际输入

- dry LiDAR rosbag：`sim_dry_baseline_001`
- 水域场景：`sim_water_20cm_001/moderate/seed_303`
- RGB 窗口：frame 129–169，共 41 帧
- 固定锚点帧：frame 149
- 视觉提示：现有时序视觉模块自动生成，无人工点击
- Ground Truth：prediction 全程未读取

## 3. 关键接口修复

仿真点云的 frame 为 `lidar_link`，现有相机几何工作在 `map`。本阶段从 rosbag 的静态 TF 解析并执行 `lidar_link → map` 变换，再生成 Ground DEM。否则点云高度和相机射线不在同一坐标系，后续水位反演没有物理意义。

S5 新增“读取已经生成的标准 S4 文件”入口。该入口不会调用旧的 `configured_depth` 反演；旧入口仍保留用于历史 MVP 对照。

## 4. 预测侧结果

- 41/41 帧候选质量门：pass
- 41/41 帧全局场景状态：complete
- 锚点 estimated water level：-0.240260 m
- 锚点 S4 面积：7.10 m²
- 锚点 S4 体积：0.667352 m³
- 锚点平均水深：9.399 cm
- 锚点最大水深：20.013 cm
- Camera reprojection IoU：0.942389
- outer boundary P95：3.606 px

旧 3 px gate 会仅因像素边界 P95 拒绝该锚点；已冻结的 C8 候选门将该指标作为诊断项而非单独否决项。本阶段没有修改旧阈值。

## 5. 现有 S5–S8 与 Agent 结果

六个现有阶段均成功：

1. S5 area/volume
2. S6 weather correction
3. S7-A deterministic forecast
4. S7-B case retrieval
5. S7-C physical constraint
6. S8 warning report

Agent 运行状态：success。

- S5 有效水域面积：6.89 m²
- S5 体积：0.666800 m³
- S5 平均水深：9.678 cm
- S5 最大水深：20.013 cm
- 仿真雨强：20 mm/h（moderate）
- 气象修正系数：1.3
- S8 结果：none
- SQLite 审计库：已生成

S5 面积小于 S4 面积，是因为原 S5 继续使用 `min_valid_depth_cm=0.5`，浅于 0.5 cm 的边缘栅格不计入面积；这不是重新配置水深。

## 6. 安全与语义

- `ground_truth_used=false`
- `authoritative=false`
- `eligible_for_real_warning=false`
- `warning_action_mode=simulation_record_only`
- 未发送短信、邮件或外部通知
- 未启动真实 Camera、LiDAR、ROS 节点或 Gazebo
- 未写入正式项目输出目录

## 7. 尚未完成

本次证明“仿真原始传感器数据可以穿过现有项目全链路”，但不代表已经真实部署：

- S6 读取的是仿真场景气象元数据，尚未接入真实天气 API；
- 当前 41 帧不足 10 分钟，S7-A 使用明确标记的静态保持历史，只验证接口，不伪造上涨趋势；
- S7-B 仍使用离线 mock 历史案例库；
- S7-C 仍是简化水量平衡模型；
- 尚未启动 HTTP API 服务或正式 Dashboard 数据接口；
- 尚未进行真实道路和真实降雨验证。

下一阶段应优先接入一个可替换、可缓存、失败可降级的真实天气 API 适配器，同时保留仿真场景元数据作为离线回退，再运行相同标准流水线。

## 8. 验证

- Python compileall：通过
- 项目全量 pytest：288 passed
- 未执行 Git commit 或 push
