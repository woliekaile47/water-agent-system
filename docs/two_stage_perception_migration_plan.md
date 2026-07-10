# water-agent-system 两阶段感知迁移与下一阶段开发计划

> 文档性质：架构分析与开发设计，不是实现结果。本文中的验收阈值均为“建议目标/待冻结门槛”，不代表已经取得实验结果。
>
> 分析日期：2026-07-10（Asia/Shanghai）

## 0. 分析基线与边界

### 0.1 实际分析对象

- Ubuntu 真实项目路径：`/home/wlkl/water_agent_ws/water_agent_system`
- 当前分支：`feature/standard-experiment-evaluation`
- 当前提交：`caec9a7c8b47d29fed2829ee3ac17276b443303e`
- `git log -1 --oneline --decorate`：`caec9a7 (HEAD -> feature/standard-experiment-evaluation, tag: v1.0-project-demo-ready, origin/main, main) Add teacher report materials for project demonstration`
- 初始 `git status --short`：无输出，工作区干净。
- 通过只读的 `git ls-remote origin refs/heads/main` 核对，GitHub `main` 当前也是 `caec9a7c8b47d29fed2829ee3ac17276b443303e`。因此本次分析不是 Windows 镜像分析，也不存在把 Windows 本地修改当成 Ubuntu 修改的问题。
- Ubuntu 环境：Ubuntu 22.04.5 LTS、ROS 2 Humble、4 vCPU、10 GiB 内存、VMware SVGA II；未发现 `gz`/`gazebo` 可执行程序，没有 NVIDIA RTX GPU。

### 0.2 本轮执行边界

本轮没有启动 LiDAR、摄像头、ROS 2 实时节点、Gazebo/Isaac Sim，也没有执行 rosbag replay；没有修改或删除已有 rosbag，没有安装依赖，没有运行会改写业务输出的 pipeline，没有执行 `git commit` 或 `git push`。唯一计划中的仓库变更是新增本文档。

### 0.3 已检查内容

重点读取或检索了：

- `run_offline_pipeline.py`
- `src/dem/build_dem.py`、`build_ground_dem.py`、`build_ground_dem_from_rosbag.py`、`build_surface_dem_from_rosbag.py`、`visualize_dem.py`
- `src/vision/extract_camera_frame.py`、`create_manual_mask.py`、`visualize_mask.py`
- `src/masks/create_dem_space_water_mask.py`、`diagnose_dem_space_mask.py`
- `src/fusion/map_mask_to_dem.py`
- `src/hydrology/invert_water_depth.py`、`invert_surface_depth.py`、`invert_boundary_waterline_depth.py`、`calculate_area_volume.py` 及可视化脚本
- `src/evaluation/evaluate_surface_depth_accuracy.py`、`diagnose_surface_depth_quality.py`、`surface_depth_quality_gate.py`
- `src/reasoning/`、`src/warning/`、`src/agent/pipeline_agent.py` 的输入依赖、结果字段和调度顺序
- `configs/*.yaml`
- `dashboard/app.py`、`dashboard/utils.py`、`dashboard/README.md`
- `README.md`、`PROJECT_STATUS.md`、`DATA_MANIFEST.md`、代表性 `data/**/metadata.json` 与 `outputs/json/*.json`
- `tests/`；当前只有 `.gitkeep`，尚无自动化测试。

## 1. 当前项目结构与真实数据流

### 1.1 当前 S1-S8

1. **S1 多模态采集**：已有 ROS 2 LiDAR + Camera 原型采集链，项目本体主要离线读取已有 rosbag。LiDAR topic 为 `/cx/lslidar_point_cloud`，Camera topic 为 `/hik_camera/image_raw`。
2. **S2 Ground DEM**：
   - S2-A `dem` 是场景高度栅格；
   - S2-B `ground_dem` 才是地面基准，读取 dry rosbag，ROI/高度过滤后按 0.1 m 栅格计算 p20，保存原始 DEM、有效掩膜、点数、最近邻插值 DEM 和 metadata。
3. **S3 Mask**：从 rosbag 抽取单帧；人工图像 polygon 生成 `manual_water_mask`。操场场景另有直接在 DEM 行列坐标上画 polygon 的 `playground_pit_water_region_mask`。
4. **S4 水深**：存在三条并行但未统一的路线：
   - S4-MVP：人工相机 mask 的统计量 + 手工 DEM 矩形 ROI → `configured_depth` 水面 → 深度图；
   - S4-real-A：降雨/积水期 LiDAR surface DEM − dry ground DEM；
   - S4-real-B：DEM-space mask 边界地面高程的截尾中位数 → 水位 → ground DEM 深度反演。
5. **S5 面积体积**：固定读取 `data/hydrology/water_depth_map.npy`、`water_depth_valid_mask.npy` 和 `outputs/json/water_depth_result.json`；当前实际对应 configured-depth MVP。
6. **S6 气象修正**：离线 mock 天气，产生雨强、短时累计雨量和修正系数。
7. **S7 混合推理**：规则预测 → mock 案例检索修正 → 简化水量平衡约束。当前上涨斜率使用 `prediction_config.yaml` 中的 mock depth history，不是连续 S4 实测序列。
8. **S8 预警**：优先读 S7-C final forecast，缺失时回退 S7-A；再生成预警 JSON、报告、图和审计日志。

### 1.2 当前三条 S4 路线的作用和问题

| 路线 | 当前作用 | 真实输入 | 当前输出 | 结论 |
|---|---|---|---|---|
| S4-MVP configured-depth | 验证 mask→DEM→depth→S5 工程链路 | 人工 mask、手工 DEM ROI、配置水深 | 通用旧文件 `water_depth_result.json`、`water_depth_map.npy` | 必须保留为历史/MVP 对照，不能称为真实测量 |
| S4-real-A surface DEM | 验证 LiDAR 水面回波直接测深 | 积水期 LiDAR rosbag、ground DEM、water mask | case-scoped surface/depth/accuracy/gate | 保留为实验对照和特定材质/深水场景的补充，不再作为新主路线 |
| S4-real-B boundary waterline | 在水面回波不稳时用边界估水位 | DEM-space mask、ground DEM | case-scoped depth、面积体积、boundary quality | 算法骨架最接近新主路线，应升级为 camera mask 驱动的主路线 |

关键事实：当前 `map_mask_to_dem.py` **不是像素级映射**。它读取相机 mask 的 shape/pixel count，但真正写入 DEM 的区域来自 `configs/roi_mapping.yaml` 中的固定矩形 `dem_water_roi`。操场 DEM-space mask 则完全绕过相机几何，是人工 DEM 行列多边形。

### 1.3 Ground DEM 的构建、存储与读取

- 基础场景：`data/dem/ground_dem.npy`、`ground_dem_valid_mask.npy`、`ground_dem_point_count.npy`、`ground_dem_interpolated.npy`、`ground_dem_metadata.json`。
- 场景级：例如 `data/dem/playground_pit/` 下相同文件名。
- 原始观测 DEM 用 `NaN` 表示未观测栅格；有效掩膜为 bool；point count 表示直接点云支持；插值采用最近邻，metadata 明确标注填充数量。
- metadata 包含 `grid_size`、`dem_shape`、`dem_roi`、来源 bag/topic、帧数、有效率、方法和输出路径。
- 当前不足：没有稳定的 `dem_id`、版本/生命周期、相机标定绑定、校验和、激活/过期状态和道路变更检测；路径有绝对路径，跨机器可移植性弱。

### 1.4 当前质量诊断与 gate

- S4-real-A diagnosis 检查有效覆盖率、点密度、已知水深误差和跨 case 重叠；gate 使用 coverage、mean error、max-depth outlier 以及诊断组合告警，输出 `quality_status` 和 `can_enter_s5_s8_warning_chain`。
- S4-real-B 内部检查边界有效点数、边界高程标准差、有效覆盖率和已知水深误差，输出 `boundary_quality_status`。
- 已有操场 6 cm 结果是 `reject`，这是已有产物，不应外推为新算法性能。
- 当前 gate 的不足：
  1. S4-real-A gate 与 S4-real-B gate 是两套实现；
  2. 实际 Agent 并不读取任一 gate；
  3. 已知水深误差只能用于仿真/标定评估，不能成为真实运行时 gate 的必需字段；
  4. 现有 `quality_status != reject` 即允许进入，早期新路线宜采用更保守的 `pass_only`；
  5. gate 没有校验结果、DEM、标定、mask 是否属于同一个 run/case/timestamp。

### 1.5 S5-S8 与 Dashboard 的实际依赖

- `area_volume` stage 会先调用 `invert_water_depth()`，因此 Agent 每次从 S5 开始都会重新生成 configured-depth MVP 深度，再计算面积体积。
- S5 硬编码读取旧通用水深文件；S7-A/S7-B/S7-C 和 S8 间接依赖 `outputs/json/water_area_volume_result.json`。
- S7-A 当前斜率来自配置中的 mock history；S8 当前不检查 S4 quality gate。
- Agent 顺序是 `area_volume → weather_correction → deterministic_forecast → case_retrieval → physical_constraint → warning_report`，没有 S1-S4、质量诊断或 gate。
- Dashboard 使用项目根目录相对路径，固定读取：
  - MVP：`outputs/json/water_depth_result.json`、`water_area_volume_result.json` 和对应图片；
  - real-A：case-scoped accuracy、quality diagnosis、quality gate JSON/报告/图片；
  - real-B：case-scoped boundary JSON、报告和 heatmap；
  - S7/S8：`final_forecast_result.json`、`warning_decision_result.json`、`warning_report.md`；
  - Agent：`agent_run_summary.json` 和 SQLite 路径。
- Dashboard 能展示 reject，但不能阻止后续链路。现有页面和路径应保留为历史对照，不应删除。

## 2. 推荐的目标架构

### 2.1 核心原则

新主链路定义为：

```text
安装/地形变化时：LiDAR dry scan → versioned ground DEM → activate baseline

降雨时：camera video
        → camera water mask（第一版可用仿真 GT）
        → image-mask-to-DEM
        → DEM-space boundary
        → per-component water-level estimation
        → depth = max(0, water_level - ground_dem)
        → quality diagnosis
        → quality gate
        → canonical current water state
        ├─ pass: S5 → S6 → S7 → S8
        └─ warning/reject: 诊断/人工复核，不进入正式预警链
```

必须先以 **仿真 camera Ground Truth mask** 跑通并验证几何映射与边界反演，再开发雨滴视觉模型。这样视觉误差和水深反演误差不会在第一轮混在一起。

### 2.2 可复用、新增和历史保留

可继续复用：

- Ground DEM 栅格化、有效掩膜、point count、插值和可视化框架；
- rosbag2 离线顺序读取器和相机消息解码；
- `invert_boundary_waterline_depth.py` 的边界形态学、稳健统计、逐栅格深度公式思路；
- `calculate_area_volume.py` 的栅格面积和体积求和；
- evaluation 的 `pass/warning/reject` 表达、报告和图表框架；
- S6-S8 的 JSON 产物、报告/审计框架；
- Agent 的 stage 执行与 SQLite 审计框架；
- Dashboard 的相对路径读取、缺失文件降级展示方式。

需要新增或重构：

- Ground DEM registry/lifecycle 与 camera calibration registry；
- 仿真 world、场景生成、GT 导出和 manifest；
- 时域雨滴特征、GT mask adapter、camera mask 生成；
- 真正的相机/DEM 几何映射及覆盖率/可见性图；
- DEM-space connected components、边界数据、per-component 水位；
- route-agnostic quality diagnosis/gate；
- `current_water_state.json` canonical manifest 和 legacy compatibility exporter；
- S7 连续 accepted-state history；
- Agent 的两阶段状态机和 gate 前置检查；
- 自动化单元/集成/回归测试。

仅作为历史/实验对照保留：

- `invert_water_depth.py` + `roi_mapping.yaml` 的 configured-depth MVP；
- `build_surface_dem_from_rosbag.py` + `invert_surface_depth.py` 的 S4-real-A；
- 人工 image polygon mask 和人工 DEM-space polygon mask；
- 现有固定输出和 Dashboard 页面。它们应标注 legacy/experimental，不能被新主路线静默覆盖。

## 3. 仿真环境设计（第一优先级）

### 3.1 平台选择

**推荐：Gazebo-first，Isaac Sim-later 的分阶段组合。**

1. 第一阶段使用 Gazebo 建立几何正确、可重复、可无复杂流体的最小闭环。当前 VM 是 ROS 2 Humble；官方默认组合是 Humble + Gazebo Fortress，但 Fortress 生命周期接近末期。新仿真资产宜优先放在独立的 Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic 环境，产出标准 ROS 2 topics/rosbag 和中立文件接口，再由现有 Humble 离线 pipeline 消费。Gazebo 官方当前也推荐新用户使用 Ubuntu 24.04、ROS 2 Jazzy 和 Gazebo Harmonic；Harmonic 支持到 2029 年 5 月。若短期必须只用当前 VM，可用 Humble 默认 Gazebo 组合做过渡，但不要引入已停止维护的 Gazebo Classic。参考：[Gazebo ROS 安装与兼容矩阵](https://gazebosim.org/docs/harmonic/ros_installation/)。
2. Isaac Sim 不作为 Phase 1 阻塞依赖。当前 VM 只有 VMware SVGA、10 GiB RAM，明显不满足当前 Isaac Sim 最低 32 GiB RAM、RTX 4080/16 GiB VRAM 级要求。待有独立 RTX 工作站/云 GPU 后，用 Isaac Sim 做更真实的材质、灯光、雨幕、镜面反射和域随机化数据生成。参考：[Isaac Sim 官方系统要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)。
3. 两个平台通过相同的场景 manifest、传感器标定、ROS topic 和 Ground Truth 文件 schema 解耦；算法不得依赖 Gazebo/Isaac 私有路径。

### 3.2 道路与水体模型

- 道路使用明确的 ENU/map 坐标；保存可审计的 heightmap/mesh 和同源数值 ground truth。
- 至少包含：正常纵横坡、一个连续低洼盆地、清晰路沿、排水口占位、可选路面接缝；路沿高度必须高于 40 cm 场景水位或在 manifest 中明确溢流边界。
- dry baseline 不包含水体；水场景使用每个连通盆地一张静态水平水面。场景标签 5/10/20/40 cm 定义为 `target_max_depth_cm = water_level - basin_min_ground_z`，不能含糊地解释为“整块路面上铺同厚水层”。
- 权威 DEM-space GT 由解析式 `mask_gt = ground_dem < water_level`、`depth_gt = max(0, water_level-ground_dem)` 生成；camera GT mask 由水面语义/instance ID 渲染并投影到相机画面。
- 第一版不模拟流动、波浪传播、排水或车轮扰动；水面是静态平面。雨滴/雨幕只作为后续视觉外观，不参与水量变化。

### 3.3 场景矩阵

| 场景 ID | 传感器模式 | 水位 | 主要用途 |
|---|---|---:|---|
| `road_basin_dry_baseline_001` | LiDAR ON；Camera 可选 | 无水 | 构建 ground DEM、标定和 dry GT |
| `road_basin_water_005_001` | LiDAR OFF；Camera ON | max 5 cm | 浅水边界/湿路混淆基线 |
| `road_basin_water_010_001` | LiDAR OFF；Camera ON | max 10 cm | 边界反演 |
| `road_basin_water_020_001` | LiDAR OFF；Camera ON | max 20 cm | 中水深/面积体积 |
| `road_basin_water_040_001` | LiDAR OFF；Camera ON | max 40 cm | 大范围/路沿约束 |

每个水场景第一批保持完全相同的道路、相机 pose、内外参和曝光；随后才增加雨强、光照、材质、相机扰动的变体。

### 3.4 传感器、时间和坐标系

固定 TF：

```text
map
└── road
    └── sensor_mount
        ├── lidar_link
        └── camera_link
            └── camera_optical_frame
```

- `map`/`road` 使用米、右手 ENU；`camera_optical_frame` 遵循 ROS optical frame（x 右、y 下、z 前）。
- 记录 `/tf_static`；每个 run 的 camera intrinsics、distortion、外参、图像尺寸、DEM frame 和 calibration ID 必须落盘。
- 所有节点/桥接使用 `/clock` 和 `use_sim_time=true`；每帧 Image、CameraInfo、GT 必须同一仿真时间基准。
- dry 模式只发布 LiDAR 点云（Camera 可用于标定但不进入降雨识别）；rain 模式由 scenario state machine 禁用 LiDAR publisher，只发布 Camera 和 GT。验收时检查雨模式 bag 中 LiDAR 消息数为 0，而不是仅让算法忽略 LiDAR。

### 3.5 ROS 2 topics 与消息

| Topic | 类型 | 说明 |
|---|---|---|
| `/clock` | `rosgraph_msgs/msg/Clock` | 仿真时钟 |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | 固定外参 |
| `/sim/lidar/points` | `sensor_msgs/msg/PointCloud2` | 仅 dry 模式 |
| `/sim/camera/image_raw` | `sensor_msgs/msg/Image` (`rgb8`/`bgr8`) | rain 视频 |
| `/sim/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | 内参/畸变 |
| `/sim/ground_truth/camera_water_mask` | `sensor_msgs/msg/Image` (`mono8`) | camera-space GT，0/255 |
| `/sim/ground_truth/dem_depth_map` | `sensor_msgs/msg/Image` (`32FC1`) | DEM 栅格深度，米；文件为权威版本 |
| `/sim/ground_truth/water_level` | `std_msgs/msg/Float64` | map frame 水面 z，米；多盆地时文件中分 component |
| `/sim/ground_truth/area_m2` | `std_msgs/msg/Float64` | GT 面积 |
| `/sim/ground_truth/volume_m3` | `std_msgs/msg/Float64` | GT 体积 |

第一版避免为仿真专门引入自定义 ROS message；完整 provenance、多个 component 和 schema version 写入 JSON manifest。后续需要在线多盆地消息时再定义自定义消息。

### 3.6 文件输出与 rosbag 命名

建议 run ID：

```text
sim_<world>_<mode>_<depthcm|dry>_<rainprofile>_<seed>_<YYYYMMDDTHHMMSSZ>
```

例如：`sim_road_basin_rain_010_static_0042_20260710T120000Z`。

rosbag 目录：`<run_id>__rosbag2`；不得用“final/new/最新”等不可排序名称。bag metadata 外另存 `simulation_manifest.json`，包含 Git commit、world hash、seed、时钟、topics、传感器状态、calibration ID、ground DEM ID、目标水位和 GT 文件校验和。

文件结构：

```text
simulation/runs/<run_id>/
├── simulation_manifest.json
├── ground_truth/
│   ├── camera_water_mask_<stamp>.png
│   ├── dem_water_mask.npy
│   ├── dem_depth_map_m.npy
│   ├── water_level.json
│   └── area_volume.json
└── <run_id>__rosbag2/...
```

与现有 pipeline 的兼容策略：提供 topic alias 配置，使现有 reader 可读取 `/sim/lidar/points` 和 `/sim/camera/image_raw`；bag 仍是 rosbag2 sqlite3/mcap 目录；输出 NPY/JSON 使用现有 DEM shape/ROI/grid conventions。不要把仿真 topic 假装成真实 `/cx/...` 或 `/hik_camera/...`，来源必须可追溯。

### 3.7 分阶段仿真验收

1. **World/geometry**：dry DEM 与解析道路高度对齐；场景 manifest、mesh hash、坐标系完整。
2. **Sensor contract**：dry bag 有 LiDAR、rain bag 无 LiDAR；Camera/CameraInfo/TF/Clock 时间一致；topic 类型固定。
3. **GT contract**：camera mask、DEM mask、water level、depth、area、volume 都存在且满足公式；不同深度场景的面积/体积单调不减（除非 manifest 明确溢流/多盆地变化）。
4. **Pipeline compatibility**：现有离线 reader 能读取模拟 bag；新增 adapter 能将 GT camera mask 送入新 S3/S4，而不读取 DEM GT 作为算法输入。
5. **Reproducibility**：同 world/config/seed 生成相同数值 GT 和同 schema manifest；图像允许记录 renderer nondeterminism 说明。

## 4. 新算法模块设计

### 4.1 雨滴动态特征提取 `src/vision/extract_rain_dynamics.py`

- 输入：带 timestamp 的连续 RGB 帧、CameraInfo、道路 ROI/有效可见区、可选 dry reference。
- 输出：时域特征张量/缓存与 metadata；建议特征包括短时亮度脉冲、局部光流/散度、圆环/径向扩散响应、纹理衰减、反射稳定度、帧间差分统计。
- 步骤：时间同步 → 去畸变/曝光归一 → 稳像 → 多时间窗差分和光流 → splash/ripple 候选 → 反射与湿路辅助特征 → 特征置信度。
- 参数：`window_frames`、`fps_min`、`flow_method`、`ripple_radius_px`、`transient_decay_ms`、`stabilization`、`exposure_normalization`、ROI。
- 异常：帧率不足、时间戳倒退、运动模糊/雨幕遮挡、相机位移时输出 invalid，不补造 mask。
- 测试：纯静态、单点短脉冲、扩散圆环、全局亮度变化、相机轻微抖动、缺帧序列；检查时域响应方向和异常标志。
- 复用：复用相机消息解码；不复用单帧 polygon 逻辑。

### 4.2 Camera water mask `src/vision/generate_camera_water_mask.py`

- 输入：动态特征、RGB clip、可选模型权重；Phase 2 输入为 GT adapter。
- 输出：`camera_water_mask.png/.npy`、per-pixel confidence、metadata。
- 步骤：特征融合/模型推理 → 阈值 → 小连通域过滤 → 时间一致性 → 输出 source type。
- 参数：模型版本、阈值、最小区域、时间投票窗口、置信度下限、是否允许 morphology。
- 异常：模型缺失、输入尺寸/标定不匹配、整幅全 0/全 1、置信度过低；标记 invalid。
- 测试：GT adapter 必须 bit-exact；预测接口做 shape/dtype/schema、空 mask、全 mask、多 component 和时间稳定性测试。
- 复用：保留 `create_manual_mask.py`，通过同一 schema 标记 `source_type=manual`；新增 `load_sim_ground_truth_mask.py` 标记 `source_type=simulation_ground_truth`。

### 4.3 Image mask → DEM `src/fusion/project_camera_mask_to_dem.py`

- 输入：camera mask/confidence、内参/畸变、`T_map_camera`、ground DEM/metadata、预计算 visibility/LUT。
- 输出：DEM-space bool mask、confidence raster、visible/covered mask、mapping diagnostics 和 overlay。
- 步骤：校验 calibration/DEM ID → 将 DEM cell center 投影到图像并检查前方、视野、遮挡/静态 visibility → 采样 mask/confidence → 多像素/多帧聚合 → 生成覆盖率。
- 参数：投影模式、采样方法、最小 pixel support、confidence threshold、visibility tolerance、边缘裁剪、畸变模型。
- 异常：TF 缺失、标定过期、图像尺寸变化、投影覆盖率低、相机位移、mask/DEM 空间不重叠；必须 invalid/reject。
- 测试：合成平面、斜坡、坑洼的已知投影；恒等/已知外参；边界像素；遮挡；shape mismatch；与仿真 DEM-space GT 计算 IoU，但 GT DEM mask只用于评估。
- 复用：复用 `map_mask_to_dem.py` 的路径/metadata/图框架；旧 `region_level_manual` 仅作 legacy adapter。

### 4.4 DEM mask 边界 `src/hydrology/extract_water_boundary.py`

- 输入：DEM-space mask、confidence、ground valid/interpolated masks。
- 输出：每个 connected component 的 inner/outer boundary、边界 cell/XY、置信度、直接观测支持率。
- 步骤：可配置轻量 morphology → 连通域 → 去小区域 → inner/outer ring → 边界排序/轮廓 → 标注 DEM support。
- 参数：4/8 邻域、最小 component 面积、closing/opening 半径、boundary band 宽度、是否接触 ROI 边缘。
- 异常：空 mask、全图 mask、组件过小、边界完全无直接 DEM 支持；不估水位。
- 测试：矩形、圆、多 component、孔洞、边缘接触、单像素、空 mask；检查边界 cell 数和拓扑。
- 复用：提取 `create_dem_space_water_mask.py`、`diagnose_dem_space_mask.py`、`invert_boundary_waterline_depth.py` 中重复的 8 邻域边界函数。

### 4.5 边界水位估计 `src/hydrology/estimate_boundary_water_level.py`

- 输入：component boundary、ground DEM、valid/direct/interpolated support、mask confidence。
- 输出：每 component 的 `water_level_m`、稳健统计、置信区间/不确定度、使用/剔除样本索引。
- 步骤：优先使用直接观测 DEM → inner/outer boundary band 配对 → 剔除低置信/突变/路沿异常 → weighted median/trimmed median 或 Huber/RANSAC → 水平性/组件一致性诊断。
- 参数：最少有效边界点、最大插值占比、trim ratio、MAD/IQR 阈值、最大边界 std、允许水位范围、component 合并距离。
- 异常：样本不足、离散过大、多峰、路沿截断、多个盆地不应共享水位；输出 unavailable 并触发 reject。
- 测试：常高边界+噪声、离群点、半边缺失、两水位 component、全插值、路沿阶跃；验证稳健统计。
- 复用：以现有 S4-real-B 截尾中位数为 baseline，拆出配置和 per-component 处理；真实运行不得使用 known depth 调参。

### 4.6 Ground DEM 水深 `src/hydrology/invert_ground_dem_depth.py`

- 输入：DEM-space mask、per-component water level、ground DEM、有效/support masks。
- 输出：float32 `depth_map_m`、wet mask、valid mask、统计与 provenance。
- 步骤：按 component 计算 `max(0, level-ground)` → 无效栅格保持 NaN/valid=false → 可选最大合理水深检查 → 汇总。
- 参数：最小湿深、最大合理水深、outside value（建议 NaN 而非 0）、component overlap policy。
- 异常：shape/ID 不一致、水位低于整个 component、异常深度、有效覆盖率低；保留诊断但不能伪装为 0 水深。
- 测试：水平/斜坡/坑洼 DEM、NaN、多个 component、负深度、单位换算和公式逐 cell 对照。
- 复用：复用三条现有 invert 的统计/绘图；主公式与 S4-real-B 一致。

### 4.7 质量诊断 `src/evaluation/diagnose_water_state_quality.py`

- 输入：mask、mapping、boundary、water level、depth、ground DEM registry、calibration registry；仿真评估时另读 GT。
- 输出：route-agnostic `water_state_quality_diagnosis.json`、报告和诊断图。
- 检查：mask confidence/时间稳定性、可见覆盖率、边界数量/离散/MAD、多峰、direct-vs-interpolated 支持、DEM/标定年龄与 ID、相机位移、有效深度覆盖率、异常最大深度、ROI 边缘截断、component 一致性。
- 仿真专属指标：mask IoU/F1、mapping IoU、水位误差、depth MAE/RMSE、area/volume error。必须标记 `evaluation_only=true`，不混入真实预测字段。
- 异常：任何输入 provenance 不一致时诊断自身为 invalid。
- 测试：逐检查项构造 pass/warning/reject fixture；缺字段和 schema version 测试。
- 复用：整合 surface diagnosis、mask diagnosis 和 boundary quality 的通用统计。

### 4.8 统一 Quality Gate `src/evaluation/water_state_quality_gate.py`

- 输入：统一 diagnosis、运行 profile、canonical draft。
- 输出：`water_state_quality_gate.json`，包含 status、checks、reasons、`can_enter_s5_s8_warning_chain`。
- 策略：硬失败优先；Phase 1-4 使用 `pass_only`，warning/reject 均不进入正式 S5-S8。后续有验证依据后才可配置 warning 是否进入 advisory chain。
- profile：`simulation_validation` 可使用 GT error；`real_operation` 禁止依赖 GT/known depth。
- 异常：gate 输入缺失、run ID/DEM ID/calibration ID 不一致、检查项未执行，一律 fail closed。
- 测试：决策表、阈值边界、未知字段、profile 混用、旧 gate adapter；确保 reject 不能通过 Agent/S8 fallback 绕过。
- 复用：保留现有 surface gate 报告格式并迁移为 route adapter。

## 5. 数据接口与 canonical current water state

### 5.1 Ground DEM

- `ground_dem.npy`：float32，单位 m，map frame，未观测为 NaN。
- `ground_dem_valid_mask.npy`：bool，直接观测有效栅格。
- `ground_dem_interpolated.npy`：float32，仅供覆盖补充；必须同时保留 direct-valid mask。
- `ground_dem_point_count.npy`：uint32。
- `ground_dem_metadata.json` 新增：`schema_version`、`ground_dem_id`、`site_id`、`frame_id`、`origin_xy_m`、`resolution_m`、`shape_rc`、`source_bag_id`、`created_at`、`activated_at`、`status(active/stale/retired)`、`valid_ratio`、`interpolated_ratio`、`calibration_compatibility`、各文件 SHA-256。

建议版本目录：`data/dem/<site_id>/<ground_dem_id>/...`，另由 registry 指向 active DEM，禁止静默覆盖。

### 5.2 Camera mask

- PNG：mono8，0=非水，255=水，便于审阅。
- NPY：bool，算法输入。
- confidence NPY：float32 `[0,1]`，预测模型必须提供；GT 可全 1 但标记 GT。
- metadata：`run_id`、`stamp_ns`、`frame_id`、`image_size`、`source_type(prediction|simulation_ground_truth|manual)`、`algorithm_id/version`、`calibration_id`、`threshold`、`valid`、`invalid_reasons`、artifact paths/hash。

### 5.3 Boundary

`water_boundary.json`：

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "ground_dem_id": "...",
  "components": [{
    "component_id": 1,
    "area_cells": 123,
    "inner_boundary_rc": [[10, 20]],
    "outer_boundary_rc": [[9, 20]],
    "boundary_xy_m": [[1.25, -0.40]],
    "direct_dem_support_ratio": 0.0,
    "mean_mask_confidence": 0.0,
    "touches_roi_edge": false
  }]
}
```

示例中的 `0.0` 是 schema 占位，不是本项目实测结果。

### 5.4 Water level

`water_level_result.json` 按 component 存：`estimated_water_level_m`、method/version、boundary sample counts、direct/interpolated counts、median/MAD/IQR/std、uncertainty、rejected samples、valid/reasons。全局单水位仅在所有 component 通过同水体一致性检查时生成。

### 5.5 Depth map 与面积体积

- `water_depth_map_m.npy`：float32，单位 m；无效为 NaN。
- `water_depth_valid_mask.npy`：bool；表示可信可计算，不等于 wet。
- `water_wet_mask.npy`：bool；建议 `depth > min_wet_depth_m`。
- 面积必须按 wet mask 计数，不能把“有效但深度为 0”的栅格算作积水面积。当前 S4-real-B 用 `valid_depth_cell_count * cell_area` 作为 area，迁移时需纠正语义。
- 体积为 `sum(depth[wet&valid] * cell_area)`。

### 5.6 Simulation Ground Truth

GT 与 prediction 分目录、分字段：

- `ground_truth/camera_water_mask_<stamp>.png`
- `ground_truth/dem_water_mask.npy`
- `ground_truth/dem_depth_map_m.npy`
- `ground_truth/water_level.json`
- `ground_truth/area_volume.json`
- `simulation_manifest.json`

算法不得把 GT 字段写入 prediction 结果冒充预测；每个评估 JSON 同时列 `prediction_artifacts` 和 `ground_truth_artifacts`。

### 5.7 质量诊断和 gate JSON

Diagnosis 顶层字段：`schema_version`、`run_id`、`route_id`、`algorithm_version`、`ground_dem_id`、`calibration_id`、`checks[]`、`metrics`、`simulation_metrics(optional/evaluation_only)`、`diagnosis_status`、`created_at`。

Gate 顶层字段：

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "profile": "real_operation",
  "quality_status": "reject",
  "can_enter_s5_s8_warning_chain": false,
  "policy": "pass_only",
  "reject_reasons": [],
  "warning_reasons": [],
  "gate_checks": [],
  "source_diagnosis": "...",
  "evaluated_at": "..."
}
```

### 5.8 Canonical result

新增 `outputs/json/current_water_state.json`，作为 S5、Agent、Dashboard 和后续 API 的唯一当前态入口：

- identity：schema/run/site/case/stamp；
- source：`sensor_mode`、`mask_source_type`、`route_id`、`algorithm_version`；
- references：ground DEM/calibration ID；
- artifacts：mask、DEM mask、boundary、water level、depth map、valid/wet masks；
- statistics：水位、max/mean/median depth、area、volume；
- quality：gate status、reasons、source gate JSON；
- admission：`formal_warning_chain_allowed`；
- truth：只允许 `null` 或独立 `evaluation_reference`，不能混入 prediction statistics。

写入应使用临时文件+原子替换，并在完成所有 artifact/hash/gate 后才发布。S5 读取 canonical 中的 artifact path 和 gate；S7/S8 读取同一 run 派生的 area-volume/current-state，不允许读时间更早的遗留文件。

### 5.9 旧输出兼容

- 迁移期提供 `export_legacy_water_depth_outputs.py`，从 **已通过 gate 的 canonical state** 投影旧 `water_depth_result.json`、`water_depth_map.npy` 和 valid mask，并写 `deprecated_alias=true`、`canonical_run_id`、`route_id`。
- `water_area_volume_result.json` 保持字段兼容，新增 run/route/quality/admission 字段。
- configured-depth 继续写自己的 legacy 文件，但不能更新 production canonical；除非运行 profile 明确是 `mvp_demo`。
- S4-real-A 和旧 boundary case 文件继续保留，不覆盖 canonical。

## 6. 建议目录树与文件影响

标记：`[N]` 新增；`[M]` 后续修改；`[R]` 继续复用；`[H]` 历史/实验对照保留。

```text
simulation/                                      [N]
├── README.md                                    [N]
├── worlds/road_basin.sdf                        [N]
├── models/road_basin/                           [N]
├── launch/two_stage_sim.launch.py               [N]
├── scenarios/
│   ├── dry_baseline.yaml                        [N]
│   ├── water_005.yaml                           [N]
│   ├── water_010.yaml                           [N]
│   ├── water_020.yaml                           [N]
│   └── water_040.yaml                           [N]
├── scripts/export_ground_truth.py               [N]
└── schemas/simulation_manifest.schema.json      [N]

configs/
├── system_config.yaml                           [M]
├── simulation_config.yaml                       [N]
├── sensor_mode_config.yaml                      [N]
├── camera_dem_calibration.yaml                  [N]
├── vision_temporal_config.yaml                  [N]
├── water_inversion_config.yaml                  [N]
├── quality_gate_config.yaml                     [N]
├── agent_config.yaml                            [M]
├── prediction_config.yaml                       [M]
├── roi_mapping.yaml                             [H]
└── surface_dem_config.yaml                      [H/R]

src/vision/
├── extract_camera_frame.py                      [R/M]
├── create_manual_mask.py                        [H/R]
├── extract_rain_dynamics.py                     [N]
├── generate_camera_water_mask.py                [N]
└── load_sim_ground_truth_mask.py                [N]

src/fusion/
├── map_mask_to_dem.py                           [H]
├── camera_dem_calibration.py                    [N]
└── project_camera_mask_to_dem.py                [N]

src/hydrology/
├── invert_water_depth.py                        [H]
├── invert_surface_depth.py                      [H]
├── invert_boundary_waterline_depth.py           [R→adapter]
├── extract_water_boundary.py                    [N]
├── estimate_boundary_water_level.py             [N]
├── invert_ground_dem_depth.py                   [N]
└── calculate_area_volume.py                     [M]

src/evaluation/
├── diagnose_surface_depth_quality.py            [H/R]
├── surface_depth_quality_gate.py                 [H/R→adapter]
├── diagnose_water_state_quality.py              [N]
├── evaluate_simulation_ground_truth.py           [N]
└── water_state_quality_gate.py                  [N]

src/agent/
└── pipeline_agent.py                            [M]

src/reasoning/                                   [R/M: canonical history]
src/warning/                                     [R/M: mandatory admission]

outputs/json/
├── current_water_state.json                     [N runtime artifact]
├── water_state_quality_diagnosis.json           [N runtime artifact]
├── water_state_quality_gate.json                [N runtime artifact]
└── existing JSONs                               [R/H compatibility]

outputs/figures/
├── camera_mask_overlay_<run_id>.png             [N]
├── mask_to_dem_overlay_<run_id>.png             [N]
├── boundary_diagnosis_<run_id>.png              [N]
├── depth_map_<run_id>.png                       [N]
└── existing figures                             [R/H]

outputs/reports/
├── simulation_validation_<run_id>.md            [N]
├── water_state_quality_report.md                [N]
└── existing reports                             [R/H]

tests/
├── unit/vision/                                  [N]
├── unit/fusion/                                  [N]
├── unit/hydrology/                               [N]
├── unit/evaluation/                              [N]
├── integration/test_gt_mask_dem_inversion.py    [N]
├── integration/test_quality_gate_blocks_s5.py   [N]
├── regression/test_legacy_outputs.py            [N]
└── fixtures/                                    [N]

dashboard/
├── app.py                                       [M]
└── utils.py                                     [R/M]

run_offline_pipeline.py                          [M]
docs/two_stage_perception_migration_plan.md       [N，本轮唯一仓库新增文件]
```

## 7. 旧路线迁移策略

### 7.1 路线标识

- `configured_depth_region_level_mvp@1`：MVP 演示，不得进入 production canonical。
- `lidar_surface_dem_difference_experimental@1`：S4-real-A 实验对照。
- `boundary_waterline_dem_mask_legacy@1`：现有 S4-real-B。
- `camera_mask_boundary_waterline@1`：新主路线。

所有结果必须同时写 `route_id`、`algorithm_version`、`mask_source_type`、`ground_dem_id`、`calibration_id`、`run_id`。

### 7.2 S4-real-A

保留，不删除。用途调整为：

- 受控深水/高回波材质下的对照；
- dry/rain LiDAR 方法研究；
- 与 camera-boundary 结果交叉诊断。

它不再要求降雨期间常开 LiDAR，也不能作为新系统部署前提。

### 7.3 S4-real-B 升级

将当前单文件逻辑拆为 boundary extraction、water-level estimator、depth inversion、quality diagnosis/gate；输入由人工 DEM-space mask 改为 camera mask 几何映射结果；支持多 component、direct/interpolated 支持度和 runtime profile。旧入口保留为 adapter，保证已有 case 报告可重现。

### 7.4 避免破坏 Dashboard/S5-S8

- 先新增 canonical，不立即删除旧文件；
- S5 先支持 `canonical|legacy` input mode，默认在新 profile 中 canonical；
- Dashboard 新增“两阶段当前态”页，现有四个 S4 页面保留并标注 legacy/experimental；
- Agent 先强制 gate，再调用 S5；
- legacy exporter 只从 gate-pass canonical 输出兼容文件；
- S8 不得因为 canonical reject 而回退到旧的 stale forecast；应输出 `monitoring_unavailable/blocked_by_quality_gate`，与“无积水/无预警”区分。

## 8. S1-S8 兼容方案

### S1：两阶段传感器状态机

- 状态：`UNINITIALIZED → DRY_BASELINE_CAPTURE → BASELINE_ACTIVE → RAIN_CAMERA_MONITORING → RECALIBRATION_REQUIRED`。
- baseline capture：LiDAR ON，Camera 可做标定；rain monitoring：LiDAR OFF，Camera ON。
- 状态切换和实际 topic 活跃性均写审计；违反互斥策略时 reject 当前 run。

### S2：Ground DEM 生命周期

- DEM versioned/immutable；通过覆盖率、噪声、地理范围、标定绑定后才 active。
- 触发重建：道路施工/沉降、路沿变化、点云变化检测、人工巡检、相机支架/传感器重新安装。
- stale/retired DEM 不得用于 production canonical；历史结果保留引用原 DEM ID。

### S3：mask source 统一

- manual、simulation GT、vision prediction 均输出同 schema；
- Phase 2 默认 GT adapter；Phase 3 才切 prediction；
- source type 必须显式，评估和 Dashboard 不可混淆。

### S4：主路线切换

- route selector 根据 profile 选择；production 目标为 `camera_mask_boundary_waterline@1`；
- configured-depth 仅 `mvp_demo`；surface DEM 仅 `experimental`；
- S4 发布 canonical draft，只有统一 gate 通过后转为 current state。

### S5：面积体积

- 核心公式可复用；修改输入解析、wet/valid 语义、run provenance 和 gate 检查；
- 拒绝未通过 gate、过期或跨 run artifact；
- 不再隐式调用 `invert_water_depth()`。

### S6：气象修正

- 算法不受感知路线直接影响；继续独立产出天气修正；
- 气象数据不能把 reject 的水深变为可用；S6 可运行但 S7/S8 必须被 admission 阻断。

### S7：连续视觉水深上涨斜率

- 用 accepted canonical states 按 timestamp 建 history store；
- 1/5/10 min 窗口使用真实时间差和质量权重；缺测/拒绝帧不当成 0；
- 样本不足时输出 slope unavailable，不回退 mock history，除非 profile 明确 `mvp_demo`。

### S8：继承 gate

- warning decision 输入加入 canonical state 和 gate；
- 检查 run ID、admission、staleness；不通过则停止正式水文预警并生成“感知质量阻断/需现场核验”的系统状态；
- audit 必须记录 gate reasons 和使用的 algorithm/DEM/calibration version。

### Agent

新调度：

```text
sensor_mode_check
→ active_ground_dem_check
→ camera_mask (GT or prediction)
→ mask_to_dem
→ boundary
→ water_level
→ depth
→ quality_diagnosis
→ quality_gate
→ publish_canonical
→ [pass only] area_volume → weather → forecast → warning → audit
```

baseline build 是独立 workflow，不应在每个 rain run 中重建。任何 stage 失败都 fail closed；Agent summary 增加 blocked stage 和 admission reason。

### Dashboard

新增显示：sensor mode、active ground DEM ID/年龄、calibration ID/年龄、mask source、route/version、camera/DEM mask、映射覆盖率、边界支持和离散、水位不确定度、gate/admission、reject reasons、accepted depth history；仿真页并排显示 prediction 与 GT。旧页面保留为对照。

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 仿真雨滴与真实雨滴域差异 | 仿真视觉指标虚高 | Phase 2 只验证几何反演；Phase 3 做域随机化；Phase 5 必须真实视频盲测 |
| 湿路与浅积水混淆 | mask 误扩张/漏检，边界偏移 | 强制时域 ripple/splash 特征；输出置信度；5 cm 单独验收；边界不稳定 reject |
| 夜间灯光、车灯、镜面反射 | 假水面 | 夜/逆光/车灯场景集；曝光归一；反射特征不能单独决策 |
| 暴雨雨幕遮挡 | 视频不可用 | visibility/blur/occlusion gate；多帧聚合；输出 unavailable 而非假 mask |
| mask 边界错误 | 水位偏差可被全区域放大 | inner/outer band、稳健估计、多 component、边界扰动敏感性测试、std/MAD gate |
| Ground DEM 过期/道路变化 | 系统性水深错误 | immutable DEM registry、定期/事件触发重建、变化检测、stale 禁用 |
| 相机或支架移动 | mask-to-DEM 全局错位 | 固定标志/道路控制点自检、外参 hash、重投影误差、移动即 recalibration required |
| 单水平水面假设失效 | 坡面流、流动/溢流、多盆地错误 | per-component 水位；流动/强坡/连通溢流场景 reject；后续扩展局部平面/水动力模型 |
| DEM 插值占比过高 | 边界高程无直接证据 | direct support ratio 纳入 gate；限制插值样本权重 |
| 坐标/单位/时间错配 | 隐蔽的大误差 | schema、frame/units、run/DEM/calibration ID 强校验；集成 fixture |
| 旧文件被误读 | reject 仍进入预警 | canonical single entry、原子发布、禁止 stale fallback、兼容 exporter 受 gate 控制 |
| 仿真代替真实验证 | 错误工程结论 | 明确 simulation-only 标签；Phase 5 前不得声明真实场景性能 |

## 10. 开发阶段与里程碑

以下命令是未来实现后的建议验收命令，本轮未执行。

### Phase 1：仿真最小闭环

- 目标：建立可重复道路/坑洼/路沿/静态水平水，完成 dry、5/10/20/40 cm 场景、传感器模式和全部 GT。
- 输入：道路几何、传感器 pose/内参、场景 YAML、seed。
- 交付物：world/models/launch、5 类场景、bags、GT、manifest、说明文档。
- 验收：`ros2 bag info <bag>` 检查 topic/type；脚本检查 rain bag LiDAR count=0、dry bag LiDAR>0；GT 公式、单调性、hash/reproducibility 全通过。
- 依赖：优先 Jazzy+Harmonic 独立仿真环境；与现有 Humble 通过标准 bag/file contract 对接。
- 暂不实现：复杂流体、真实雨滴识别、车辆、排水、后端 API。

### Phase 2：Ground Truth mask + DEM 水深反演

- 目标：只用 camera GT mask 验证 mask→DEM→boundary→water level→depth→area/volume。
- 输入：Phase 1 dry LiDAR bag、rain camera GT mask、CameraInfo/TF、GT（仅评估端）。
- 交付物：GT adapter、几何映射、边界、水位、深度、统一 diagnosis/gate、simulation validation report。
- 验收：`pytest tests/unit/fusion tests/unit/hydrology tests/integration/test_gt_mask_dem_inversion.py`；所有场景满足公式和 schema。建议在开发前冻结数值门槛，例如 mapping IoU、water-level/depth/area/volume error；报告实际值，未达标不得进入 Phase 3。建议目标不是既有结果。
- 依赖：Phase 1、active ground DEM 和标定。
- 暂不实现：视觉预测 mask、真实雨滴模型、正式 S5-S8。

### Phase 3：雨滴视觉 mask MVP

- 目标：用时域 splash/ripple/反射特征产生 prediction mask，并与 GT mask 独立评估。
- 输入：仿真视频+GT，随后加入少量真实标注视频用于域差异观察。
- 交付物：dynamic features、mask model/rules、confidence、时域可视化、训练/评估拆分 manifest。
- 验收：固定 test split，先冻结 IoU/F1、boundary F-score、时间稳定性和 invalid detection 门槛再测试；GT adapter 回归结果不得变化。
- 依赖：Phase 2 已证明反演模块正确。
- 暂不实现：宣称真实道路泛化、端到端联合调参、S8 正式预警。

### Phase 4：质量门控与 S5-S8 集成

- 目标：canonical current state、统一 pass-only gate、S5/S7/S8/Agent/Dashboard 接通。
- 输入：Phase 2/3 prediction artifacts、DEM/calibration registry、天气数据。
- 交付物：canonical publisher、legacy exporter、route-agnostic gate、accepted history、Agent 新 DAG、Dashboard 新页。
- 验收：`pytest tests/integration/test_quality_gate_blocks_s5.py tests/regression/test_legacy_outputs.py`；人为构造 reject 时 S5-S8 不运行且无 stale fallback；pass case 的旧 Dashboard 核心页面仍可读。
- 依赖：Phase 2 schema 稳定；Phase 3 可选，GT route 也能先集成。
- 暂不实现：数据库/API 服务化、自动紧急处置。

### Phase 5：真实视频验证

- 目标：验证真实雨滴、湿路、浅水、夜间/车灯、暴雨遮挡和相机稳定性；校准仿真到真实域差异。
- 输入：合规采集并人工标注的真实视频、同步人工水位/面积参考、active ground DEM。
- 交付物：数据 manifest、盲测报告、failure taxonomy、门槛调整记录、模型卡。
- 验收：按预注册 split 和冻结阈值生成报告；所有指标明确样本量/场景，不把仿真指标当真实指标；失败场景应被 gate 拒绝。
- 依赖：Phase 4 的 provenance/gate。
- 暂不实现：未经验证的全天候正式部署声明。

### Phase 6：后端数据库与 API

- 目标：保存 versioned DEM、calibration、canonical states、quality、warning 和 artifacts；对外提供查询而非直接读散落文件。
- 输入：Phase 4 schema、Phase 5 运行数据和审计需求。
- 交付物：DB migration、artifact store、只读 API、鉴权/保留策略、Dashboard API adapter。
- 验收：schema migration/回滚、幂等写、run provenance、reject 不可发布、API contract 和审计测试。
- 依赖：canonical schema 在 Phase 4 稳定。
- 暂不实现：自动控制交通设施、未经授权的外部告警发布。

## 11. 建议的近期最小交付物

下一次开发只做以下最小集合：

1. 一个 `road_basin` Gazebo world；
2. dry + 5/10/20/40 cm 静态水平水场景；
3. dry LiDAR bag、rain Camera bags、TF/Clock/CameraInfo；
4. camera GT mask、water level、DEM GT depth/area/volume 和 manifest；
5. 由 dry LiDAR 构建的 versioned ground DEM；
6. GT camera mask → 几何映射 → boundary → water level → depth 的离线验证报告；
7. 不含任何雨滴视觉模型，不接正式 S5-S8，避免两个未知模块同时进入。

只有该闭环通过，才开始视觉 mask MVP。

## 12. 决策摘要

- 新主路线：**dry LiDAR ground DEM + rain Camera mask + boundary waterline inversion**。
- 仿真策略：**Gazebo 几何闭环优先，Isaac Sim 仅在具备 RTX 环境后用于视觉真实性与域随机化**。
- 安全策略：**canonical current water state + fail-closed unified quality gate + pass-only admission**。
- S4-real-A、configured-depth、人工 mask 全部保留为实验/历史对照，不删除、不静默覆盖。
- S5 核心公式可复用，但必须解除硬编码和隐式 configured-depth 调用；S7 改用连续 accepted states；S8 必须继承 gate。
- 第一验收对象是“GT camera mask + DEM 边界反演”，不是雨滴视觉识别。
