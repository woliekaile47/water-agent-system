# Data Manifest

This project keeps small configuration, source code, JSON summaries,
reports, and figure artifacts in Git. It does not keep raw rosbags or
large generated arrays in Git.

## Project Data Directories

| Path | Purpose | Git policy |
|---|---|---|
| `configs/` | Offline pipeline configuration files | tracked |
| `data/dem/` | DEM metadata and local DEM arrays | metadata tracked; `.npy` ignored |
| `data/masks/` | Manual mask metadata and local mask arrays/images | metadata tracked; `.npy` ignored |
| `data/fusion/` | Mask-to-DEM mapping metadata and local masks | JSON tracked; `.npy` ignored |
| `data/hydrology/` | Water depth, area, and volume results | JSON tracked; `.npy` ignored |
| `data/meteorology/` | Offline mock weather correction results | JSON tracked |
| `data/reasoning/` | S7-A/B/C reasoning outputs | JSON tracked |
| `data/warnings/` | S8 warning decision JSON | JSON tracked |
| `data/audit_logs/` | JSONL audit logs | tracked when small |
| `data/db/` | SQLite runtime audit database | database files ignored |
| `outputs/json/` | Latest MVP JSON outputs | tracked when small |
| `outputs/figures/` | Latest MVP figures | tracked when small |
| `outputs/reports/` | Markdown reports | tracked |

## Ignored Large File Types

The repository intentionally ignores these generated or raw data types:

- raw rosbag data: `*.db3`, `*.mcap`, `*.bag`, `*.bag/`
- point cloud exports: `*.pcd`, `*.ply`
- large generated arrays: `*.npy`, `*.npz`
- SQLite runtime databases: `data/db/*.db`, `data/db/*.sqlite`,
  `data/db/*.sqlite3`
- ROS workspace outputs: `build/`, `install/`, `log/`

## Required Tracked Inputs

These files should exist in the repository for the offline MVP demo:

- `configs/roi_mapping.yaml`
- `configs/weather_config.yaml`
- `configs/prediction_config.yaml`
- `configs/case_retrieval_config.yaml`
- `configs/physical_constraint_config.yaml`
- `configs/warning_config.yaml`
- `configs/agent_config.yaml`
- `data/dem/ground_dem_metadata.json`
- `data/masks/manual_water_mask_metadata.json`
- `data/cases/mock_historical_cases.json`

## Ignored But Locally Required Inputs

These files are ignored by `.gitignore` because they are generated arrays,
but the current offline demo expects them to be present locally:

- `data/dem/ground_dem.npy`
- `data/dem/ground_dem_interpolated.npy`
- `data/dem/ground_dem_valid_mask.npy`
- `data/fusion/water_region_mask.npy`
- `data/masks/manual_water_mask.npy`
- `data/hydrology/water_depth_map.npy`
- `data/hydrology/water_depth_valid_mask.npy`

If these files are missing, restore them from the local data directory or a
backup, or regenerate earlier stages before running the Agent pipeline.

## Reproducibility Notes

The current repository does not store rosbag files or large `.npy` arrays.
To reproduce the full experiment from raw data, restore those local files
from backup or the original `~/water_agent_data` location first.

The files under `outputs/json/` and `outputs/reports/` can be used to inspect
the latest offline MVP run result.
