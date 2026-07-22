# Phase 2D-C-10C：比赛合成旁路演示版发布说明

## 发布标识

- Tag：`v1.1-competition-synthetic-shadow-demo`
- 发布性质：比赛用合成数据旁路演示版
- 正式预警：未启用
- 真实道路验证：未完成

## 已完成能力

1. 无雨道路 Ground DEM 基准及可重复仿真场景；
2. 动态降雨 Camera 序列；
3. 不读取 Ground Truth 的时序证据与自动 SAM 2 提示；
4. SAM 2 连续帧双向传播；
5. Camera 外岸线、ray/DEM 求交和稳健水位估计；
6. 水深、Camera 可见面积和体积；
7. `camera_visible_estimate` 与 `global_scene_estimate` 可观测性语义；
8. 合成数据候选 quality gate；
9. canonical shadow state、S5-S8旁路契约、独立审计和Dashboard；
10. 仿真道路专用比赛页面和一键离线启动入口。

## 验收证据

- 全量自动测试：258项通过；
- C9安全不变量：22/22通过；
- C9故障注入：9/9被正确拒绝；
- Seed 303：12序列、492帧；
- 候选门控：361 pass、131 reject；
- 合成确认集中，361个pass帧水位误差均小于3 cm，未观察到false pass；
- 5/10/20/40 cm比赛页面均完成人工显示验收；
- 5 cm `reject/unavailable`显示为红色；
- 40 cm `partial`显示为黄色，不冒充完整全域估计。

## 演示命令

```bash
cd /home/wlkl/water_agent_ws/water_agent_system
bash scripts/run_phase2d_c10_competition_demo.sh
```

浏览器打开`http://localhost:8501`。

## 明确限制

- 当前结果只证明合成仿真环境下的研究闭环；
- 尚未完成真实道路、夜间、车灯、暴雨雨幕和长期运行验证；
- 5 cm浅水仍是主要视觉困难；
- 单Camera无法安全恢复视野外独立盆地；
- 动态降雨仿真与真实雨滴存在域差异；
- 结果保持`authoritative=false`、`eligible_for_downstream=false`；
- 不接入正式S5-S8预警动作。

## 后续路线

1. 设备可用后优先验证真实0–5 cm浅水、湿路面和无水负样本；
2. 使用独立测尺或水位计获得真实水深；
3. 根据真实旁路数据重新评估自动提示、SAM 2和quality gate；
4. 通过多Camera标定与统一map坐标系扩大可观测范围；
5. 真实旁路运行稳定后，另行评审正式S5-S8启用条件。
