# Surface Depth Quality Diagnosis

This diagnosis is for S4-real offline LiDAR surface DEM depth inversion.

Important conclusions:

- Results are diagnostic and should not be treated as final engineering accuracy.
- The 13cm dormitory scene is clearly overestimated when included in this report.
- Playground pit data is used as an additional controlled S4-real validation scene.
- Low valid-depth coverage indicates that results should be interpreted cautiously.

## Per-case Diagnosis

| Case | Known cm | Mean cm | Median cm | Max cm | Mean error cm | Coverage | Depth cells | Surface cells | High outliers | Low outliers | Warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| playground_pit_water_sim_6cm_001 | 6.00 | 39.01 | 36.27 | 149.02 | 33.01 | 0.1401 | 1960 | 3511 | 1455 | 0 | high_error, low_coverage |

## Cross-case Comparison

- cross-case comparison requires at least two cases

## Interpretation

The diagnosis intentionally does not tune ROI, thresholds, or DEM settings to fit known depths. High error or low coverage should be treated as a real quality signal for the current data and calibration setup.
