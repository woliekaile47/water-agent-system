# Surface DEM Depth Accuracy Report

This report evaluates offline LiDAR surface DEM difference depth for the dormitory simulation scenes.

The current evaluation is based on dormitory simulated 13cm / 39cm water-depth scenes. It does not represent final engineering accuracy. It is mainly used to validate the upgrade from configured_depth to offline LiDAR surface difference.

| Case | Known depth cm | Mean cm | Median cm | Max cm | Mean error cm | Median error cm | Valid ratio | Coverage | Accuracy | Data quality |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| playground_pit_water_sim_6cm_001 | 6.00 | 39.01 | 36.27 | 149.02 | 33.01 | 30.27 | 0.1401 | coverage_warning | accuracy_warning | ok |
| water_sim_13cm_001 | 13.00 | 34.72 | 35.07 | 60.51 | 21.72 | 22.07 | 0.2000 | coverage_warning | accuracy_warning | ok |
| water_sim_39cm_001 | 39.00 | 39.14 | 45.90 | 60.51 | 0.14 | 6.90 | 0.2000 | coverage_warning | ok | ok |

## Notes

- Low valid depth ratio indicates low data quality and should be interpreted cautiously.
- The current method does not use `configured_depth`.
- Accuracy depends on calibration, ROI mapping, point density, and LiDAR returns from the water/surface target.
