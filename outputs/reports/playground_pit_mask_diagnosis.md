# Playground Pit DEM-space Mask Diagnosis

This report diagnoses the DEM-space water-region mask used by S4-real-B boundary waterline inversion.

The script does not tune the polygon to match known depth. It only reports mask geometry, boundary height stability, and point-density diagnostics for manual refinement.

## Summary

- mask_shape: [1722, 708]
- mask_cell_count: 2880
- mask_area_m2: 28.8000
- row range: 1220 - 1309
- col range: 122 - 153
- boundary_cell_count: 240
- boundary_valid_cell_count: 68
- boundary_valid_ratio: 0.2833
- boundary height median/std m: 0.3695 / 0.4976

## Boundary Outliers

- deviation > 10 cm: 31
- deviation > 20 cm: 25
- deviation > 50 cm: 19

## Point Count

- mask point_count min/mean/median/max: 0.00 / 246.16 / 0.00 / 5423.00
- boundary point_count min/mean/median/max: 0.00 / 142.75 / 0.00 / 5423.00

## Manual Refinement Guidance

- If boundary height std is high, inspect the DEM and point-count overlays and shrink or reshape `refined_polygon_points_rc` around the true pit water region.
- If many boundary points are outliers, the polygon likely crosses slope, wall, sparse LiDAR returns, or non-water terrain.
- Recreate the mask with `--use-refined` only after manual review; do not tune the polygon solely to match `known_depth_cm`.
