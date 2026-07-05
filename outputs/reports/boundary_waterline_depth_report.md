# S4-real-B Boundary Waterline Depth Report

S4-real-B estimates water level from the water-region boundary on the ground DEM, then computes depth inside the water region.

This method is useful when direct LiDAR water-surface returns are unstable, but it depends on the correctness of the water mask boundary and ground DEM.

| Case | Quality | Boundary cells | Water level m | Boundary median m | Boundary std cm | Mean cm | Median cm | Max cm | Known cm | Mean error cm | Area m2 | Volume m3 | Mask source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| playground_pit_water_sim_6cm_001 | reject | 68 | 0.3695 | 0.3695 | 49.7572 | 57.5033 | 58.9438 | 141.9659 | 6.0000 | 51.5033 | 12.9000 | 7.4179 | playground_pit_dem_space_mask |

## Notes

- If `mask_source` is `existing_water_region_mask_mvp`, the result is an MVP fallback. A playground_pit-specific water mask should be configured before formal validation.
- S4-real-B does not tune parameters to known depth and does not modify rosbag data.
- Quality status `reject` means the result should be kept for diagnosis only.
