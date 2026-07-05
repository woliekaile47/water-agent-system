# S4-real Surface Depth Quality Gate Report

This report records whether S4-real offline LiDAR surface-depth results are allowed to enter downstream S5-S8 warning stages.

S4-real is still in an experimental stage. Rejected results are preserved as diagnostic artifacts only.

| Case | Quality status | Known cm | Mean cm | Max cm | Mean error cm | Valid ratio | Can enter S5-S8 | Main reasons |
|---|---|---:|---:|---:|---:|---:|---|---|
| playground_pit_water_sim_6cm_001 | reject | 6.00 | 39.01 | 149.02 | 33.01 | 0.1401 | False | low_coverage: valid_depth_ratio_in_water_region=0.1401 < 0.15; high_mean_error: abs(mean_error_cm)=33.01 cm > 20.00 cm; extreme_max_depth_outlier: max_depth_cm=149.02 cm > known_depth_cm + 50.00 cm (56.00 cm); combined_quality_failure: high_error_warning=True and coverage_warning=True |

## Interpretation

- `playground_pit_water_sim_6cm_001` is judged as `reject` when the current metrics show low coverage, high mean error, and extreme maximum-depth outlier behavior.
- Rejected S4-real results are saved for diagnosis only and must not enter the formal S5-S8 warning chain.
- The dormitory 39cm controlled scene indicates the offline surface DEM chain is feasible, but shallow-water playground scenes need further calibration and quality control.
- The quality gate does not tune parameters, alter rosbags, or force measured depths toward known depths.
