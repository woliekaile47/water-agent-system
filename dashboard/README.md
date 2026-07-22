# Streamlit Dashboard

This dashboard（可视化看板） displays offline results from
`water_agent_system`. It does not start live LiDAR, camera, ROS nodes, or
rosbag replay.

Run:

```bash
cd ~/water_agent_ws/water_agent_system
streamlit run dashboard/app.py
```

If Streamlit prints a local URL, open it in a browser. Typical addresses
are:

- `http://localhost:8501`
- `http://<vm-ip>:8501`

The dashboard shows:

- project overview and S1-S8 pipeline,
- S4-MVP configured-depth results,
- S4-real-A surface DEM direct inversion,
- S4-real quality gate,
- S4-real-B boundary waterline inversion,
- DEM-space mask diagnosis,
- S7/S8 warning results,
- Agent summary and SQLite audit database path.
- a simulation-road-only competition page built from the frozen Seed 303
  dynamic-rain Camera, SAM 2 mask, DEM inversion, and candidate-gate outputs.

Prepare the competition display snapshot before opening its page:

```bash
python3 scripts/build_phase2d_c10_competition_demo.py
```

For the competition, use the dedicated entry point so historical dormitory
comparison pages are not part of the presentation navigation:

```bash
bash scripts/run_phase2d_c10_competition_demo.sh
```

The competition page rejects dormitory/cardboard, manual-prompt, Ground Truth,
and non-simulation input paths. It remains display-only, non-authoritative, and
ineligible for the formal warning chain.

Missing files are shown as warnings instead of crashing the page.
