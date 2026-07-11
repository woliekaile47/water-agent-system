# Phase 2D-B-2A 典型序列岸线可视化审计

## 1. 目的与约束

本阶段基于 Phase 2D-B-1 的 23 条 water geometry diagnostics，审计极端 P95、近 3 px 阈值以及 Camera mask IoU 较高但 geometry reject 的典型序列，重点区分：

- 局部轮廓异常与整体边界偏移；
- 错误小连通域、孔洞和大范围碎片化；
- 视觉岸线问题与水位/DEM 重建问题；
- unknown 是否直接影响主要岸线；
- 为什么面积型 Camera IoU 较高时，尾部敏感的 boundary P95 仍可能 reject。

本阶段只读取已有 prediction、evaluation 和 geometry diagnostics 输出：没有重新运行 temporal prediction，没有修改算法、阈值、gate、GT 或正式评价域，没有接入 S5–S8、数据库或 Dashboard，所有结果继续 `eligible_for_downstream=false`。

## 2. 案例选择

选择规则与数量：

| 分组 | 条件 | 数量 |
|---|---|---:|
| Extreme | boundary P95 > 10 px | 5 |
| Near threshold | 3 < boundary P95 ≤ 4 px | 5 |
| High Camera IoU reject | Camera IoU ≥ 0.8 且 geometry reject | 8 |

组间允许重叠，去重后共审计 15 条：

1. `sim_water_10cm_001/heavy/seed_202`
2. `sim_water_10cm_001/light/seed_41`
3. `sim_water_10cm_001/moderate/seed_42`
4. `sim_water_10cm_001/moderate/seed_101`
5. `sim_water_20cm_001/heavy/seed_43`
6. `sim_water_20cm_001/light/seed_41`
7. `sim_water_20cm_001/moderate/seed_42`
8. `sim_water_20cm_001/moderate/seed_203`
9. `sim_water_40cm_001/heavy/seed_43`
10. `sim_water_40cm_001/heavy/seed_112`
11. `sim_water_40cm_001/heavy/seed_202`
12. `sim_water_40cm_001/heavy/seed_204`
13. `sim_water_40cm_001/light/seed_41`
14. `sim_water_40cm_001/moderate/seed_42`
15. `sim_water_5cm_001/moderate/seed_42`

## 3. 每案例可视化与指标

每条案例生成一张六面板 PNG：

1. predicted Camera water mask；
2. unknown mask；
3. reprojected Camera mask；
4. observed/reprojected trusted boundaries 叠加；
5. 双向最近边界距离热图；
6. 已有 temporal proxy 曲线及数据可用性声明。

结构化审计同时记录：

- observed/reprojected 连通域数量和面积；
- 非最大连通域及 ≤100 px 小连通域数量；
- 边界距离 >3 px、>10 px 的比例；
- >10 px 尾部像素的空间包围盒范围；
- unknown 接触 raw water boundary 的像素比例；
- observed/reprojected water pixel area ratio；
- Camera IoU、reprojection IoU、水位误差、depth MAE 和 gate reasons。

## 4. Temporal 曲线的真实可用性

已有 artifacts 没有保存逐帧 water masks，也没有逐帧 shorelines。因此无法可靠恢复：

```text
temporal water-mask area curve: unavailable
temporal shoreline stability curve: unavailable
```

15/15 均只有以下可用信息：

- `candidate_count_by_frame`；
- `water_ripple` track observations，可形成逐帧 observation-area sum；
- 单个 `water_mask_time_stability` 标量。

图中的橙线明确标记为 `water-track area proxy`：它是被分类为 `water_ripple` 的 track observation area 之和，track 之间可能重叠，不能解释为 water-mask 像素面积，也不能用于证明 mask 随帧扩张或收缩。候选曲线和该代理曲线可用于发现时域证据突发与不稳定，但不能替代真正的逐帧岸线序列。

## 5. P95 > 10 px 的五条极端案例

| Case | P95 | Camera IoU | >10 px 边界比例 | Area ratio | Observed components | Stability | 可视化审计结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 10 cm heavy/202 | 14.204 | 0.636 | 11.90% | 1.156 | 1 | 0.099 | 单连通域，但 observed 轮廓较窄且有局部凹陷，reprojected 更平滑、更大；不是整体刚性平移 |
| 20 cm light/41 | 14.879 | 0.536 | 15.33% | 0.872 | 2 | 0.425 | observed 被分成两个较大的连通块，reprojected 为单一平滑区域；属于视觉区域断裂和形状不一致 |
| 20 cm moderate/203 | 13.441 | 0.673 | 11.44% | 1.131 | 1 | 0.445 | 单连通域外轮廓存在凹陷/局部缺口，reprojected 填充并扩张；属于局部视觉边界与重建形状共同差异 |
| 40 cm heavy/204 | 37.828 | 0.742 | 16.37% | 1.170 | 4 | 0.008 | mask 内部出现大孔洞、分裂结构和两个 ≤100 px 小组件；reprojected 是完整平滑区域，极端 P95 主要来自错误内部边界与拓扑破碎 |
| 40 cm light/41 | 31.006 | 0.429 | 43.21% | 1.030 | 10 | 0.373 | 大范围碎片化、多个孔洞和 6 个小组件；错误不是少数异常点，而是分布式拓扑失真 |

五条极端案例不是同一种“固定偏移”：

- 40 cm light/41 是明显分布式碎片化，>10 px 尾部包围范围约占图像 11.66%；
- 40 cm heavy/204 是内部孔洞/错误内岸线与小组件；
- 20 cm light/41 是两个大连通块；
- 10 cm heavy/202 和 20 cm moderate/203 仍是单连通域，但 observed/reprojected 的面积和局部凹陷不同。

因此，极端 P95 主要反映视觉 mask 拓扑和局部形状错误，而不是少数孤立边界点或统一方向平移。

## 6. 3–4 px 近阈值案例

| Case | P95 | Camera IoU | >3 px 边界比例 | >10 px 比例 | Area ratio | Components | Stability |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10 cm light/41 | 3.606 | 0.643 | 8.98% | 0 | 1.007 | 1 | 0.653 |
| 10 cm moderate/42 | 3.606 | 0.887 | 6.26% | 0 | 1.015 | 1 | 0.647 |
| 10 cm moderate/101 | 3.162 | 0.763 | 6.14% | 0 | 1.010 | 1 | 0.714 |
| 20 cm heavy/43 | 4.000 | 0.875 | 11.07% | 0 | 1.006 | 1 | 0.849 |
| 5 cm moderate/42 | 3.606 | 0.891 | 5.95% | 0 | 1.034 | 1 | 0.682 |

这五条均为单连通域，没有 >10 px 尾部，observed/reprojected 面积比只差约 0.6%–3.4%。图上主体轮廓高度重合，超阈值主要集中在局部轮廓离散、窄端或小幅凹凸差异。

它们与极端案例存在明确结构差异：近阈值案例没有碎片化、孔洞链或大范围形状断裂，更接近离散岸线/水位轮廓的小范围差异。这为以后在独立 calibration set 上重新研究 3 px 阈值提供案例，但本阶段没有修改阈值。

## 7. Camera IoU ≥ 0.8 但 geometry reject

8 条高 Camera IoU 案例全部是单连通域，observed/reprojected 面积比约为 1.006–1.033。其 Camera IoU 高，是因为大面积内部区域重合；boundary P95 对少量局部轮廓尾部更敏感：

- 5/10 cm moderate/42：P95 3.606 px，仅约 6% 边界超过 3 px；
- 20 cm heavy/43：P95 4 px，主体轮廓重合良好；
- 20 cm moderate/42：P95 5 px，局部窄端和曲率差异；
- 40 cm heavy/43、112、202 和 moderate/42：P95 5–6.325 px，内部面积仍高度重合，但局部凹口、轮廓平滑度或水位生成的 DEM 外轮廓不同；其中部分还保留不可观测第二盆地的全局 reject 语义。

所以“Camera IoU 高但 P95 reject”并不矛盾：IoU 是面积主导指标，局部边界错误对总面积影响很小；P95 则专门放大最差 5% 的边界距离。

## 8. 小连通域、孔洞与 unknown

15 条中：

- observed water mask 多于一个连通域：3 条；
- 存在 ≤100 px 非最大小连通域：2 条；
- unknown 接触 raw water shoreline：0 条。

多连通域集中在极端案例：20 cm light/41、40 cm heavy/204、40 cm light/41。小组件集中在两个 40 cm 极端案例。

所有 15 条的 unknown 均未直接接触 raw observed water boundary，因此本批典型案例的岸线差异不是由 unknown 切断主要岸线造成。这与 Phase 2D-B-1 的 known/full-domain P95 完全一致结论相互支持。

## 9. 应在视觉岸线阶段修复的问题

建议下一阶段优先在视觉输出侧研究，但暂不在本阶段实现：

1. **外岸线与内部孔洞分离**：P95 不应让明显视觉伪孔洞自动成为可信零水深岸线；内部孔洞需要独立证据和语义。
2. **小组件持续性门控**：对 ≤100 px 或相对主组件很小的水域，要求跨帧持续、空间邻接和水纹证据，避免瞬时 rain-impact 轨迹形成孤立水域。
3. **跨帧 signed-distance consensus**：未来保存逐帧 masks 后，在 signed distance field 上做时间中位数/稳健融合，而不是只对最终二值 mask 做形态学操作。
4. **拓扑稳定性指标**：增加连通域数、孔洞数、Euler characteristic 和外轮廓持续性；将突变作为 visual gate 诊断输入。
5. **岸线不确定带**：输出确定 water、确定 non-water 和 boundary-uncertain band，避免把不稳定凹口作为精确岸线。
6. **谨慎的 hole filling/closing**：只对跨帧无支持、面积受限的伪孔洞使用，不能无条件填洞，否则会掩盖真实道路孤岛或露出区域。
7. **保存逐帧诊断 artifacts**：至少保存降采样 per-frame mask、外岸线或 signed-distance summary，才能真实评价 mask 扩张/收缩和 shoreline stability curve。

## 10. 属于水位或 DEM 重建的问题

以下模式更适合在视觉拓扑稳定后由几何侧继续诊断：

- observed 为单一、平滑、高 IoU 区域，但 reprojected 外轮廓整体略大或略小；
- 面积比持续偏离 1，且水位误差同步增加；
- 高 Camera IoU 下仅局部曲率/窄端不匹配；
- 40 cm 可见主盆地轮廓良好，但全局仍受不可观测第二盆地限制。

建议使用固定 Camera mask 的轻量 water-level sweep，绘制水位变化与 reprojection boundary distance 的关系，区分“水位标量误差”与“DEM 栅格/连通重建形状误差”。该 sweep 应保持 GT 隔离并只用于诊断，不自动调整正式阈值。

## 11. 当前结论

1. P95>10 的极端错误多数不是整体边界平移，而是孔洞、碎片化、多连通域或局部形状/面积不一致。
2. 近阈值案例结构干净，主体轮廓高度重合，仅少量局部边界超过 3 px。
3. 高 Camera IoU 无法保证 P95 低，因为面积重合对局部边界尾部不敏感。
4. unknown 没有接触这 15 条案例的主要岸线，不是本批错误来源。
5. 真正的逐帧 mask area 和 shoreline stability 曲线无法从当前 artifacts 恢复；water-track area 仅是可视化代理。
6. 优先修复视觉岸线拓扑稳定性，再对单连通、高 IoU 案例做水位/DEM 形状诊断；当前不修改 gate 或 3 px 阈值。

## 12. 输出

输出目录：

```text
outputs/synthetic_visual_to_depth_integration/shoreline_case_audit/
```

包含：

- `case_selection.json`
- `case_audit_summary.csv`
- `case_audit_summary.json`
- `cases/<case_id>__<rain_level>__seed_<seed>.png`，共 15 张

输出沿用 `outputs/synthetic_visual_to_depth_integration/` 的忽略规则，不加入 Git。
