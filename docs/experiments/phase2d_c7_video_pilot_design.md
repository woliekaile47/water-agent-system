# Phase 2D-C-7-1：SAM 2 连续帧传播 pilot 设计

## 目的

本阶段验证单个冻结自动 prompt 能否通过 SAM 2 video predictor 在连续帧中传播，并输出不依赖 GT 的 mask 面积曲线与相邻帧 IoU。重点判断浅水强雨的单帧过分割是否具有跨帧不稳定性。

## 固定 pilot

选择 3 条 seed 302 序列：5 cm heavy 作为浅水强雨失败型，20 cm moderate 作为稳定参考，40 cm light 作为 partial coverage 型。每条使用原始 frame 79–119，共 41 帧，anchor 固定为已冻结的 frame 99 自动 prompt。

选择在传播前冻结，不浏览视频效果，不读取 Camera mask、水位、DEM、depth、面积或体积 GT。pilot 配置保存在 `configs/phase2d_c7_video_pilot.yaml`。

## 本地 SAM 2 API 核验

WSL SAM 2 官方仓库 commit 为 `2b90b9f`。当前 API 使用：

1. `build_sam2_video_predictor(config, checkpoint, device)`；
2. `predictor.init_state(video_path=...)`；
3. 在 anchor local frame 调用 `add_new_points_or_box`；
4. 从 anchor 分别执行 forward 和 reverse `propagate_in_video`；
5. 以 logit > 0 保存每帧二值 unknown candidate mask。

官方目录加载器只读取按整数命名的 JPEG，因此 runner 将 PNG 窗口复制转换为 `00000.jpg` 等文件，固定 quality 95、4:4:4 subsampling，不改变图像尺寸或提示坐标。源 PNG 和生成 JPEG 均记录 SHA-256。该有损编码差异必须在结论中披露。

## 输出语义

每帧保存 raw mask、mask 面积、相邻帧 IoU 和哈希；序列汇总面积变异系数、相邻 IoU 分布、GPU 峰值和耗时。所有输出保持：

- `semantic_label=unknown_candidate`
- `authoritative=false`
- `ground_truth_used=false`
- `eligible_for_downstream=false`

本阶段不修改现有单帧 prompt、SAM2 mask、quality gate 或 S5-S8，也不使用传播结果自动修正 anchor。
