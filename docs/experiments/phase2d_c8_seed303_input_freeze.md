# Phase 2D-C-8-3A：seed 303 独立确认输入冻结

## 1. 目的

本阶段为 C8 候选 quality gate 创建一组未参与规则设计的独立确认输入。在查看任何 seed 303 RGB 或评价结果前，样本选择已固定为：

- 4 个水深 case：5、10、20、40 cm；
- 3 个雨强：light、moderate、heavy；
- 统一新 seed：303；
- 每个序列 200 帧、20 FPS；
- anchor frame：149；
- 连续确认窗口：frame 129–169，共 41 帧；
- 总计 12 个序列、2,400 张生成 RGB，后续窗口共 492 帧。

没有浏览多帧后选择效果较好的样本，也没有读取任何 evaluation 输出。

## 2. 生成方式

复用已有 `scripts/generate_dynamic_rain_sequences.py` 和固定的 `configs/dynamic_rain_visual_simulation.yaml`，逐 case、逐雨强显式传入 `--seed 303 --no-preview`。

生成器从既有 dry rosbag 只读提取基础 Camera RGB，并使用仿真 Camera water mask 合成干地飞溅与水面涟漪。该 mask 属于数据生成端 Ground Truth；后续 prediction 不得读取或继承它。生成过程未启动 ROS 节点、Gazebo、Camera、LiDAR 或真实设备。

## 3. 生成结果

12/12 序列均满足：

- frame count = 200；
- first/last = frame 000000 / 000199；
- generation quality status = pass；
- anchor 和 window 文件完整；
- manifest、sequence、anchor 和 41 帧 window SHA-256 已写入冻结矩阵；
- 生成数据由既有 `data/simulation_dynamic/*` 规则排除，不进入 Git。

各雨强的事件数在四个水深 case 中由相同 seed 和固定规则确定：

| 雨强 | dry events | water events |
|---|---:|---:|
| light | 11 | 4 |
| moderate | 22 | 18 |
| heavy | 36 | 44 |

12 个序列占用约 1.34 GB 本地空间，仅保留在 VMware 数据目录中。

## 4. 冻结配置

冻结清单位于 `configs/phase2d_c8_seed303_confirmation_matrix.yaml`，同时锁定：

- 动态生成配置 SHA-256；
- C6C 固定自动提示配置 SHA-256；
- C8-2 候选 gate 配置 SHA-256；
- 每个样本的 anchor、window、manifest 和 sequence SHA-256。

冻结配置不包含 Camera/DEM mask GT、真实水位、depth、area、volume 或 evaluation 指标。

## 5. 当前边界与下一步

本阶段没有运行自动提示、SAM 2、视频传播、岸线提取、几何 prediction 或独立 GT evaluation，`prediction_started=false`。

下一小阶段 C8-3B 将先验证全部冻结哈希，再使用固定 C6C 自动提示规则在 12 个 anchor 上生成提示并运行一次 SAM 2；之后对固定 41 帧窗口做视频传播和 prediction-side 几何。所有 prediction 冻结后，C8-3C 才允许独立读取 GT 评价候选 gate。
