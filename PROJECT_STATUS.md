# Project Status

| Stage | Status | Current implementation |
|---|---|---|
| S1 | Complete | ROS2 LiDAR + camera prototype acquisition chain |
| S2 | Complete | Ground DEM baseline |
| S3 | Complete | Manual polygon water candidate mask |
| S4 | Complete | Region-level mask-to-DEM mapping + MVP water depth |
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

## MVP Simulation Boundary

The current result is an offline engineering MVP. It uses
`configured_mvp_simulation`, `offline_mock_weather`,
`offline_mock_depth_history`, `offline_mock_case_library`, and
`simplified_water_balance_mvp`. These are not final real-world emergency
dispatch or hydrodynamic forecast results.
