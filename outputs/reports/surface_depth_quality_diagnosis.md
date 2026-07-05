# Surface Depth Quality Diagnosis

This diagnosis is for S4-real offline LiDAR surface DEM depth inversion.

Important conclusions:

- The 13cm dormitory scene is clearly overestimated in the current result.
- Current dormitory data is used to validate the S4-real pipeline and diagnose quality; it is not a final accuracy conclusion.
- An open playground scene should be recollected with a new dry baseline and water cases for a more formal S4-real validation.
- `valid_depth_ratio=0.20` indicates low coverage and the results should be interpreted cautiously.

## Per-case Diagnosis

| Case | Known cm | Mean cm | Median cm | Max cm | Mean error cm | Coverage | Depth cells | Surface cells | High outliers | Low outliers | Warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| water_sim_13cm_001 | 13.00 | 34.72 | 35.07 | 60.51 | 21.72 | 0.2000 | 22 | 57 | 11 | 1 | high_error, low_coverage |
| water_sim_39cm_001 | 39.00 | 39.14 | 45.90 | 60.51 | 0.14 | 0.2000 | 22 | 49 | 4 | 6 | low_coverage |

## Cross-case Comparison

- overlapping_valid_cell_count: 22
- overlap_ratio: 1.0000
- same_cell_depth_difference_mean_cm: 4.42
- same_cell_depth_difference_median_cm: 0.00
- cells_valid_only_in_13cm: 0
- cells_valid_only_in_39cm: 0

## Interpretation

The diagnosis intentionally does not tune ROI, thresholds, or DEM settings to fit the 13cm target. The high 13cm error should be treated as a real quality signal for the current dormitory data and calibration setup.
