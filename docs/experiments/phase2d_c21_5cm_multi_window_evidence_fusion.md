# Phase 2D-C-21：5 cm 多时间窗口证据融合

## 目的

5 cm 浅水在单个约2秒窗口中容易被飞溅、涟漪和反光干扰。C21不删除这些干扰，
而是在60秒视频的多个固定短窗口中寻找反复出现的浅水核心，再自动生成SAM 2的
box、正点和负点。

本阶段始终保持：

- prediction不读取Ground Truth；
- 不人工挑选窗口或补点；
- 不按held-out结果修改参数；
- prompt未通过时不运行SAM 2；
- `authoritative=false`、`eligible_for_downstream=false`。

## 代码入口

| 作用 | 文件 |
|---|---|
| 多窗口融合算法 | `src/vision/temporal_multi_window_evidence_fusion.py` |
| 命令行入口 | `scripts/fuse_temporal_multi_window_evidence.py` |
| 统一规则 | `configs/temporal_multi_window_evidence_fusion.yaml` |
| C21/C17旧Prompt规则 | `configs/temporal_sam2_prompt_corroborated.yaml` |
| seed310数据配置 | `configs/phase2d_c21_5cm_60s_seed310_dynamic.yaml` |
| seed310实验协议 | `configs/phase2d_c21_5cm_60s_seed310_pipeline.yaml` |
| 测试 | `tests/test_temporal_multi_window_evidence_fusion.py` |

旧C19/C20单窗口算法没有被覆盖或删除。

C22后续的可行三点组合方法使用独立配置文件，不改变本文件记录的C21旧Prompt规则。

## 固定协议

- case：`sim_water_5cm_001`
- rain level：`moderate`
- 视频：60秒、20 FPS、1,201帧
- 13个候选窗口，每窗41帧，步长5秒
- 最少3个支持窗口、至少15秒跨度
- 安全正点要求：至少3个
- 正点必须离组件边界至少3 px
- 正点之间至少间隔10 px
- probability至少0.50
- 跨窗口支持率至少2/3

## 实现审查与正确性修复

第一次seed309运行后，代码审查发现旧实现把“初步空间簇”的窗口数和时间跨度
误当成“最终融合核心”的真实支持。4 px容差投票也可能生成原始窗口未实际预测的
像素。因此seed309结果被标记为：

`invalidated_after_post_run_code_review`

seed309输出保留用于审计，但不能作为最终held-out证据，也没有重新运行。

修复内容：

1. 最终核心只能落在原始窗口mask的联合区域内；
2. 选出最终核心后，逐窗口重新计算实际覆盖；
3. 使用实际支持窗口重算数量、持续时间和代表窗口；
4. 校验1,201帧且编号从0开始连续；
5. 增加配置取值范围校验；
6. manifest记录五份输入配置SHA-256；
7. 移除未使用的13份完整中间prediction，降低内存占用；
8. 增加链式聚类、容差不造水、缺帧、prompt拒绝等回归测试。

修复后规则只在开发seed308上检查，不读取GT。结果为5个最终核心支持窗口、跨度
55秒、4个安全正点，融合和prompt均通过。随后代码与配置冻结，再生成全新seed310。

## Seed310正式held-out结果

seed310只运行一次。动态数据生成完整：1,201帧，generation quality为`pass`。

### 多窗口融合

| 指标 | 结果 |
|---|---:|
| 候选窗口 | 13 |
| 可参与融合窗口 | 7 |
| 最终核心实际支持窗口 | 7 |
| 实际支持时间跨度 | 40 s |
| required support count | 4 |
| maximum support count | 6 |
| 最终核心面积 | 391 px |
| 最终核心bbox | `[330,166,30,19]` |
| 歧义核心数 | 0 |
| 自动负点 | 8 |
| 自动安全正点 | 2 |
| prompt状态 | `reject` |
| SAM 2是否允许运行 | false |

唯一拒绝原因：`insufficient_safe_positive_points`。

系统按设计停止，没有运行SAM 2、岸线、几何、水位、depth或GT evaluation。
因此seed310没有水位误差指标；不能把“无输出”写成“水位误差超过3 cm”。

### 冻结哈希

| 文件 | SHA-256 |
|---|---|
| `multi_window_fusion.json` | `306f7cd228cd4e20f3dd2ed77b5bd6c302d4c94dc8c0bbe264b1a3bf3dd3e993` |
| `automatic_prompt.json` | `f696b7757c0ba8e3e8ec6f34081feb36a070d67d854fc5a1f102418fb954152c` |
| `prompt_diagnostics.json` | `2ec8ea0887ee98f52bb222a8bca72295bb51b0acd69b04bb921935d5ebdfc5b9` |
| `fusion_manifest.json` | `3e186d6f0b67fa3d424e19c57e35e5ac69b15bafbb5697ba490cee55e729d800` |

## 为什么只有2个正点

冻结数组的只读审计结果：

| 筛选阶段 | 候选像素数 |
|---|---:|
| 融合主组件 | 391 |
| 离边界至少3 px | 209 |
| probability至少0.50 | 391 |
| 跨窗口支持率至少2/3 | 239 |
| 同时满足边界和概率 | 209 |
| 同时满足全部三项 | 175 |

现有算法按顺序贪心选择：

1. 首点`[348,174]`，边界距离8.197 px；
2. 第二点`[336,175]`，距首点12.042 px；
3. 选择这两点后，剩余173个安全候选中，没有一个同时距两点达到10 px；
4. 最好的第三点是`[355,172]`，最小间距仅7.280 px，所以被spacing规则拒绝。

但在相同175个候选、相同3 px边界要求和10 px间距要求下，确实存在合法三点组，
例如：

- `[346,169]`
- `[336,174]`
- `[355,174]`

还存在最小两两间距约11.705 px的三点组。因此：

- 数据本身并非不足；
- “至少3点”和“10 px间距”本身并未被证明确实过严；
- 直接原因是当前贪心顺序先选中心最深点、再选最远点，没有先保证三点整体可行；
- 这是自动提示选点策略的实现局限，不是GT、SAM 2、ray–DEM或水位模块错误。

## 正式结论

seed310是正式、有效的held-out失败结果：

- 多窗口浅水核心检测通过；
- 跨窗口持续证据充分；
- 自动负点充分；
- 自动正点组合策略失败；
- 系统安全停止，未形成authoritative measurement；
- 未读取seed310 GT；
- 未重新生成、人工补点、降低阈值或继续尝试其他seed。

因此C21证明了“60秒多窗口融合能找到5 cm重复浅水核心”，但还没有证明“自动提示
能跨新seed稳定触发SAM 2”。

## 建议下一阶段：C22

C22只修改正点组合方法，不降低任何安全阈值：

1. 将seed310转为下一阶段的development failure case；
2. 在175个安全候选上先搜索满足最少3点的确定性可行集合；
3. 优先保证三点两两间距≥10 px，再按边界距离、概率和支持率排序；
4. 达到3点后，再以maximin方式尝试扩展到目标5点；
5. 加入“合法三点集合存在时不得只返回两点”的单元测试；
6. 只用冻结的seed310数组做prompt-only开发，不重跑60秒prediction；
7. 规则冻结后，再用一个全新seed做一次held-out；
8. held-out通过后才运行SAM 2、几何和独立GT评价。

这一步不应降低`min_positive_points=3`、`min_positive_spacing_px=10`，也不应使用GT
选择点。

## 验证状态

- `python3 -m compileall src tests scripts`：通过；
- 主测试：`393 passed`；
- 仿真测试：`14 passed`；
- 没有seed310相关进程运行；
- 没有启动ROS节点、Gazebo、Camera、LiDAR或RTSP；
- 没有执行commit或push。
