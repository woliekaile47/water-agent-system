# S8 Waterlogging Warning Report - MVP

This report is generated from MVP simulation data and is not final real emergency dispatch advice.

The final warning is based on S7-C final forecast when available.

## Current Status

- Forecast source: **S7C_final_forecast**
- S7 pipeline used: S7A, S7B, S7C
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

## S7 Three-Layer Hybrid Reasoning Results

### S7-A Deterministic Forecast

- k_forecast: 3.0326 cm/min
- source: offline_mock_depth_history
- 5 min: 59.81 cm, warning orange
- 15 min: 90.13 cm, warning orange
- 30 min: 100.00 cm, warning orange
- 60 min: 100.00 cm, warning orange

### S7-B Case Retrieval Correction

- case source: offline_mock_case_library
- top retrieved cases: mock_case_001_moderate_rain_fast_rise (0.7523), mock_case_002_moderate_rain_compact_area (0.7513), mock_case_006_moderate_rain_wide_area (0.4090)
- median bias cm: 5min=-3.50, 15min=-6.00, 30min=-10.00, 60min=-16.00
- 5 min corrected: 56.31 cm (deterministic 59.81 cm), warning orange
- 15 min corrected: 84.13 cm (deterministic 90.13 cm), warning orange
- 30 min corrected: 90.00 cm (deterministic 100.00 cm), warning orange
- 60 min corrected: 84.00 cm (deterministic 100.00 cm), warning orange

### S7-C Physical Constraint Check

- method: simplified_water_balance_mvp
- tolerance ratio: 0.1500
- 5 min: corrected 56.31 cm -> adjusted 48.62 cm, check adjusted, confidence low
- 15 min: corrected 84.13 cm -> adjusted 58.83 cm, check adjusted, confidence low
- 30 min: corrected 90.00 cm -> adjusted 56.21 cm, check adjusted, confidence low
- 60 min: corrected 84.00 cm -> adjusted 42.00 cm, check adjusted, confidence low
- physical confidence summary: `{"overall_physical_confidence": "low", "counts": {"high": 0, "medium": 0, "low": 4}}`
- S7-C overall warning level: orange

### Final Forecast Used By S8

- 5 min: 48.62 cm, warning yellow, confidence low
- 15 min: 58.83 cm, warning orange, confidence low
- 30 min: 56.21 cm, warning orange, confidence low
- 60 min: 42.00 cm, warning yellow, confidence low

## S8 Warning Decision

- Overall warning level: **orange**
- Time to blue threshold: 5.00 min
- Time to yellow threshold: 5.00 min
- Time to orange threshold: 15.00 min
- Action suggestion: Orange warning: high risk. Recommend immediate on-site verification, traffic restriction, or temporary closure. First orange forecast horizon: 15 min.

## MVP Note

S8 MVP uses S7-C final forecast when available, with fallback to S7-A deterministic forecast. Current results are based on configured_mvp_simulation, offline_mock_weather, offline_mock_depth_history, offline_mock_case_library, and simplified_water_balance_mvp. It is not final real emergency dispatch advice.

S7-C uses simplified MVP water balance to validate physical constraint checking. It is not a full hydrodynamic model or final real forecast.

## Input Files

- `/home/wlkl/water_agent_ws/water_agent_system/outputs/json/final_forecast_result.json`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/json/physical_constraint_result.json`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/json/case_retrieval_result.json`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/json/corrected_forecast_result.json`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/json/deterministic_forecast_result.json`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/json/water_area_volume_result.json`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/json/weather_correction_result.json`

## Output Files

- `/home/wlkl/water_agent_ws/water_agent_system/data/warnings/warning_decision_result.json`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/json/warning_decision_result.json`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/reports/warning_report.md`
- `/home/wlkl/water_agent_ws/water_agent_system/data/audit_logs/warning_audit_log.jsonl`
- `/home/wlkl/water_agent_ws/water_agent_system/outputs/figures/warning_summary.png`
