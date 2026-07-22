# Project Status

## Current Two-Stage Perception Status

Release target: `v1.1-competition-synthetic-shadow-demo`.

| Phase | Status | Current implementation |
|---|---|---|
| Phase 1A | Complete | Gazebo Fortress road/Camera environment, deterministic geometry PointCloud2, Ground Truth and reproducible dry/5/10/20/40 cm scenarios |
| Phase 2A | Complete baseline | Camera mask projected to ground DEM with independent Ground Truth evaluation |
| Phase 2B | Complete | Water-surface-aware shoreline ray/DEM intersection, robust water-level estimation, basin reconstruction and observability semantics |
| Phase 2C | Research support complete | Dynamic rainfall visual simulation and interpretable temporal baselines retained for data generation, automatic prompt evidence and regression |
| Phase 2D-C6 | Complete | Fixed Ground-Truth-free temporal rules automatically generate SAM 2 box, positive points and negative points |
| Phase 2D-C7 | Complete | SAM 2 bidirectional video propagation plus per-frame shoreline/DEM geometry and temporal stability diagnostics |
| Phase 2D-C8 | Complete synthetic confirmation | Frozen candidate gate confirmed on Seed 303: 12 sequences, 492 frames, 361 pass and 131 reject |
| Phase 2D-C9 | Complete shadow integration | Canonical water state, S5-S8 shadow envelopes, independent shadow audit/API payload and Dashboard monitoring |
| Phase 2D-C10 | Competition demo accepted | Dedicated simulation-road-only Dashboard, source allowlist, one-command launcher and manual visual acceptance |
| Real-road validation | Not complete | Devices are unavailable during the school holiday; no production or real-rain accuracy claim |
| Formal warning deployment | Not complete | `authoritative=false` and `eligible_for_downstream=false`; warning actions remain disabled |

### Current Evidence

- Seed 303 confirmation: 492 frames across 5/10/20/40 cm and three rain levels.
- Candidate gate: 361 pass, 131 reject.
- All 361 passed frames were within the 3 cm water-level-error research target;
  no false pass was observed in that synthetic confirmation set.
- 5 cm remains conservative and difficult; only 9/123 frames passed, while
  5 cm heavy passed 0/41.
- 40 cm Camera-visible estimates passed, but global status remains `partial`
  where an independent basin is outside Camera coverage.
- C9 end-to-end shadow acceptance passed 22/22 invariants and 9/9 injected
  fault checks without modifying formal S5-S8 files or the formal audit DB.

### Competition Demo Boundary

The dedicated demo reads only frozen simulation-road Camera, SAM 2, geometry,
and candidate-gate artifacts. It rejects Ground Truth, manual-prompt,
dormitory/cardboard and non-simulation source paths. It does not start ROS,
Gazebo, Camera, LiDAR, RTSP, the formal Agent, or warning actions.

## Legacy S1-S8 Status

| Stage | Status | Current implementation |
|---|---|---|
| S1 | Complete | ROS2 LiDAR + camera prototype acquisition chain |
| S2 | Complete | Ground DEM baseline |
| S3 | Complete | Manual polygon water candidate mask |
| S3-playground | Initial implementation | DEM-space polygon water mask for playground_pit, matching the playground ground DEM shape |
| S4 | Complete | Region-level mask-to-DEM mapping + MVP water depth |
| S4-real | Experimental, quality-gated | Offline LiDAR surface DEM difference depth inversion; 39cm dormitory case shows the chain is feasible, while 13cm dormitory and playground 6cm cases expose overestimation / low-coverage quality issues |
| S4-real-B | Initial implementation | Boundary-based waterline inversion from water mask boundary and ground DEM; intended for shallow water or unstable LiDAR water-surface returns |
| S5 | Complete | Area and volume calculation |
| S6 | Complete | Offline mock weather correction |
| S7-A | Complete | Deterministic rule engine |
| S7-B | Complete | Offline mock case retrieval correction |
| S7-C | Complete | Simplified physical constraint check |
| S8 | Complete | Warning decision, report output, and audit log |
| Agent | Complete | Offline pipeline Agent + SQLite audit summary |
| Realtime | Not complete | Future rosbag replay / live device integration |
| LLM Agent | Not complete | Future report interpretation layer |

## Current Offline Agent Chain

```text
area_volume -> weather_correction -> deterministic_forecast ->
case_retrieval -> physical_constraint -> warning_report
```

## S4-real Surface DEM Depth Inversion

S4-real adds an offline LiDAR surface DEM difference method:

```text
current_surface_dem - ground_dem_interpolated = surface_difference_depth
```

It is intended to replace the earlier `configured_mvp_simulation` depth
source in future pipeline versions. The current implementation is still
offline only; it reads existing rosbags and does not start real-time
LiDAR, camera, ROS nodes, or rosbag replay.

Current dormitory validation shows that the 39cm scene is close to the
manual known depth, while the 13cm scene is clearly overestimated. This
is a diagnostic result, not a tuned accuracy claim. A playground open
scene should be recollected with a new dry baseline and water cases for
more formal S4-real validation.

S4-real now includes a quality gate before downstream warning use. The
gate labels each case as `pass`, `warning`, or `reject` based on valid
depth coverage, known-depth error when available, extreme outlier depth,
and diagnosis warnings. Results that fail the gate are retained for
diagnosis only and must not enter the formal S5-S8 warning chain.

## S4-real-B Boundary Waterline Inversion

S4-real-B estimates a water level from the water-region boundary on the
ground DEM, then computes:

```text
depth = max(0, estimated_water_level - ground_dem)
```

This branch is designed for shallow-water or unstable LiDAR water-surface
return scenes. It can compute depth, area, volume, and a boundary quality
status, but it depends strongly on the water mask boundary and ground DEM
quality.

For `playground_pit`, S4-real-B now requires a dedicated DEM-space water
mask at:

- `data/masks/playground_pit_water_region_mask.npy`

The old dormitory mask has a different grid shape and must not be used
for the playground ground DEM. If the playground mask is missing or has
the wrong shape, S4-real-B refuses to continue instead of using a fallback.

The first playground DEM-space mask has the correct shape but still needs
manual refinement. Mask diagnosis now reports boundary height stability,
boundary outliers, and point-count coverage so `refined_polygon_points_rc`
can be adjusted based on geometry and data coverage, not tuned to a known
water-depth value.

## Playground Pit 6cm Scene

Collected playground pit rosbags:

- `playground_pit_dry_baseline_001`
- `playground_pit_water_sim_6cm_001`

The `playground_pit` scene is a controlled pit water-depth validation
scene with `known_depth_cm = 6.0`. It is used as the second S4-real
validation scene. The 6cm water case must use the ground DEM built from
`playground_pit_dry_baseline_001`; it must not use the dormitory ground
DEM or another playground baseline.

The current playground 6cm S4-real result is clearly overestimated and
has low valid-depth coverage. Its quality gate status is expected to be
`reject`, so it is preserved as a diagnostic result and is not allowed
to enter S5-S8 warning generation.

## MVP Simulation Boundary

The current result is an offline engineering MVP. It uses
`configured_mvp_simulation`, `offline_mock_weather`,
`offline_mock_depth_history`, `offline_mock_case_library`, and
`simplified_water_balance_mvp`. These are not final real-world emergency
dispatch or hydrodynamic forecast results.
