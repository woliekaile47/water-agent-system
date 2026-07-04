# S8 Waterlogging Warning Report - MVP

This report is generated from MVP simulation data and is not final real emergency dispatch advice.

## Current Status

- Overall warning level: **orange**
- Current mean depth: 44.64 cm
- Current max depth: 62.95 cm
- Confidence: 0.75 (mvp_rule_based)

## S5 Area And Volume

- Water area: 0.7200 m2
- Water volume: 0.3214 m3

## S6 Weather Correction

- Rainfall intensity: 20.00 mm/h
- Rainfall level: moderate_rain
- Weather correction factor: 1.30

## S7-A Forecast

- k_forecast: 3.0326 cm/min
- Time to blue threshold: 0.00 min
- Time to yellow threshold: 0.00 min
- Time to orange threshold: 1.77 min

- 5 min: 59.81 cm, warning orange
- 15 min: 90.13 cm, warning orange
- 30 min: 100.00 cm, warning orange
- 60 min: 100.00 cm, warning orange

## S8 Warning Decision

- Overall warning level: **orange**
- Action suggestion: Orange warning: high risk. Recommend immediate on-site verification, traffic restriction, or temporary closure. Orange threshold may be reached very soon. Immediate action is recommended.

## MVP Note

S8 MVP uses deterministic forecast output from configured_mvp_simulation, offline_mock_weather, and offline_mock_depth_history. It is not final real emergency dispatch advice.

S7-A uses configured_mvp_simulation depth, offline_mock_weather, and offline_mock_depth_history to validate deterministic forecasting pipeline. It is not final real short-term waterlogging forecast.

## Input Files

- `/home/wlkl/water_agent_ws/water_agent_system/outputs/json/deterministic_forecast_result.json`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/json/water_area_volume_result.json`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/json/weather_correction_result.json`

## Output Files

- `/home/wlkl/water_agent_ws/water_agent_system/data/warnings/warning_decision_result.json`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/json/warning_decision_result.json`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/reports/warning_report.md`
- `/home/wlkl/water_agent_ws/water_agent_system/data/audit_logs/warning_audit_log.jsonl`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/figures/warning_summary.png`
