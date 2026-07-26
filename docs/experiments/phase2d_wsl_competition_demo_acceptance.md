# Phase 2D WSL-only Competition Demo Acceptance

## Purpose

This stage moves the competition demonstration path to WSL and displays the
existing end-to-end simulation results directly. The Dashboard no longer needs
the VMware runtime or an SSH/SCP transfer from VMware.

The displayed pipeline is the real project path used by the completed WSL
simulation matrix:

`simulation RGB/SAM2 propagated mask -> shoreline geometry -> ground DEM depth inversion -> quality gate -> S5-S8/Agent`

It is not a separate competition-only prediction algorithm.

## Frozen input

- Matrix ID: `wsl_migration_matrix_20260726_01`
- Runtime: WSL local
- Cases: 5, 10, 20 and 40 cm, moderate rain
- Source summary:
  `outputs/phase2d_c15_multiscenario_acceptance/wsl_migration_matrix_20260726_01/acceptance_summary.json`
- Dashboard snapshot:
  `outputs/phase2d_wsl_competition_demo_snapshot/competition_demo_snapshot.json`

The snapshot builder reads completed prediction and Agent artifacts only. It
does not read Ground Truth and does not rerun SAM2 or the prediction pipeline.

## Displayed outcomes

| Case | Camera-visible status | Global-scene status | Agent status |
| --- | --- | --- | --- |
| 5 cm | reject | unavailable | blocked by quality gate |
| 10 cm | pass | complete | success |
| 20 cm | pass | complete | success |
| 40 cm | pass | partial | blocked by quality gate |

The 40 cm result remains a camera-visible estimate. The second basin is outside
the camera-observable region, so the Dashboard does not present the visible
area as a complete global-scene estimate.

## Launch

Reuse the accepted matrix and start quickly:

```bash
cd /home/wlkl/water_agent_ws/water_agent_system
source /opt/ros/humble/setup.bash
bash scripts/run_wsl_competition_demo.sh \
  --matrix-id wsl_migration_matrix_20260726_01
```

Then open `http://localhost:8501/` in the Windows browser.

To intentionally rerun the complete 5/10/20/40 cm WSL matrix before display:

```bash
bash scripts/run_wsl_competition_demo.sh --refresh
```

The refresh mode is slower and is not required for a normal competition
presentation.

## Acceptance evidence

- Snapshot generation: passed, four cases produced.
- Referenced RGB, mask, reprojection and temporal plot files: all present.
- Streamlit health endpoint: `ok`.
- Dashboard root HTTP response: `200`.
- Headless page render: one title, one case selector and sixteen metric cards;
  no page exception.
- Project tests: 317 passed.
- Simulation tests: 14 passed.
- `git diff --check`: passed.

The WSL Python environment emits non-fatal warnings from optional distro
`numexpr`/`bottleneck` extensions compiled against NumPy 1.x while NumPy 2.x is
active. Pandas, the Dashboard and all acceptance checks still complete. No
dependency was changed during this stage.

## Safety boundary

- `ground_truth_used=false`
- `authoritative=false`
- `eligible_for_downstream=false`
- `eligible_for_real_warning=false`
- no real Camera, LiDAR or RTSP input
- no ROS or Gazebo node is started by the quick display path
- no external API call or real warning action

The Dashboard is a read-only competition presentation of synthetic-road
results. It must not be described as a deployed real-road warning system.
