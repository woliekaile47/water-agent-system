# “预鉴”项目中文团队交接指南

## 1. 这是什么项目

项目目标是监测低洼道路积水。无雨时由LiDAR建立道路高程地图；降雨时摄像头识别积水区域，系统结合水域岸线和道路高程地图，计算水位、水深、面积和体积。不可信结果由质量门控拒绝。

当前完成的是合成仿真环境中的研究闭环和比赛旁路演示，尚未完成真实道路长期验证，也没有启用正式预警。

## 2. 统一使用的冻结版本

- 仓库：`https://github.com/woliekaile47/water-agent-system`
- 标签：`v1.1-competition-synthetic-shadow-demo`
- Commit：`54238c7`
- 版本性质：比赛合成数据旁路演示版

写软著、论文或答辩材料时必须同时记录标签和commit，避免不同同学使用不同版本。

## 3. 系统流程

```text
无雨LiDAR -> ground DEM道路高程基准
降雨Camera视频 -> 时序证据 -> 自动SAM 2提示
-> SAM 2水域候选和视频传播 -> 外岸线
-> 岸线射线与ground DEM求交 -> 估计水面高度
-> 水深、面积、体积 -> quality gate
-> canonical shadow state -> 旁路Dashboard
```

### 简单术语解释

- `ground DEM`：没有积水时的道路高低地图。
- `water mask`：图像中哪些像素属于候选积水区域。
- `SAM 2`：根据提示分割目标区域的第三方视觉模型，本项目用它分割水域候选。
- `shoreline`：候选水域最外层岸线。
- `quality gate`：判断结果是否可信；不可信时拒绝。
- `camera_visible_estimate`：只代表相机看得到的区域。
- `global_scene_estimate`：代表当前场景全局结果完整。
- `shadow`：旁路运行，只观察和记录，不触发正式预警。

## 4. 软著材料负责人看这里

### 建议作为原创软件说明的模块

- `src/vision/`：时序证据生成自动SAM 2提示；
- `src/fusion/`：Camera岸线、ray/DEM映射和水域重建；
- `src/hydrology/`：水位与水深计算；
- `src/evaluation/`：质量门控、诊断和独立评价；
- `src/integration/`：统一水状态和旁路接口；
- `src/agent/`、`src/database/`、`src/api/`：旁路监控、审计和只读数据契约；
- `dashboard/`：比赛演示和旁路监控页面；
- `scripts/`：离线调度、验证和演示入口。

### 不能写成本项目原创的内容

- SAM 2官方模型、代码和权重；
- PyTorch、OpenCV、NumPy、Pandas、Matplotlib；
- ROS 2、Gazebo Fortress、ros_gz；
- Streamlit框架本身；
- 其他第三方依赖。

软著申请前由团队确认软件名称、版本号、著作权人、开发人员、完成日期及学校成果归属要求。代码材料只使用冻结标签，不使用WSL中的SAM 2官方仓库或自动生成输出代替原创代码。

## 5. 论文负责人看这里

### 推荐论文主线

1. 两阶段感知：无雨LiDAR地形基准＋降雨视觉感知；
2. 自动时序证据生成SAM 2提示；
3. SAM 2视频传播获得连续水域候选；
4. Camera岸线映射到ground DEM并反演水位；
5. 可观测性语义和安全质量门控；
6. 多水深、多雨强、多seed合成验证。

### 推荐先阅读的文档

- `docs/two_stage_perception_migration_plan.md`：总体迁移方案；
- `docs/experiments/phase2d_c6a_automatic_prompt_interface_design.md`：自动提示接口；
- `docs/experiments/phase2d_c7_video_pilot_results.md`：SAM 2视频传播；
- `docs/experiments/phase2d_c8_3c_seed303_gate_confirmation.md`：Seed 303门控确认；
- `docs/experiments/phase2d_c9d_end_to_end_acceptance.md`：旁路端到端安全验收；
- `docs/experiments/phase2d_c10c_competition_release.md`：比赛版本结论和限制。

### 论文可以使用的当前结论

- Seed 303包含12条序列、492帧；
- 候选门控361帧pass、131帧reject；
- 在该合成确认集中，361个pass帧水位误差均小于3 cm；
- 未观察到false pass；
- 5 cm浅水是主要困难；
- 单Camera对独立不可见盆地只能输出`partial`；
- 结果是合成仿真结论，不能直接代替真实道路结论。

### Ground Truth隔离原则

Prediction模块不能读取Camera GT mask、DEM GT mask、真实水位、真实depth、面积或体积。Ground Truth只能由独立evaluation读取。论文中必须区分预测结果、Ground Truth、预测侧质量状态和是否允许进入下游。

## 6. 数据如何分享

GitHub只保存代码、配置、测试和实验说明。PNG、NPY、JSON、CSV、SQLite和大量仿真帧由`.gitignore`排除，因此论文复现还需要独立云盘数据包。

建议云盘分成：

```text
paper_evidence_package/
├── version_and_environment/
├── frozen_configs/
├── seed303_gate_results/
├── c9_acceptance/
├── selected_figures/
├── experiment_documents/
└── manifest_sha256.json
```

云盘使用只读链接，并记录压缩包SHA-256。不要分享SSH私钥、密码、个人隐私数据或许可不允许再分发的模型权重。

## 7. 比赛演示

```bash
cd /home/wlkl/water_agent_ws/water_agent_system
bash scripts/run_phase2d_c10_competition_demo.sh
```

页面固定展示5/10/20/40 cm仿真道路中雨场景：5 cm如实拒绝，10/20 cm完整通过，40 cm为相机可见范围的partial结果。页面不展示宿舍纸箱历史素材。

## 8. 当前还没完成什么

- 真实道路和真实降雨长期验证；
- 夜间、车灯、镜面反射和暴雨雨幕系统验证；
- 真实0–5 cm浅水性能确认；
- 多摄像头统一标定和全域覆盖；
- 正式S5–S8预警启用；
- 生产级服务、权限、运维和故障恢复。

这些限制必须保留在软著功能说明、论文结论和比赛答辩中，不能为了展示效果删除。
