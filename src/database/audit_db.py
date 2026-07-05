#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite audit database for water_agent_system."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                agent_name TEXT,
                mode TEXT,
                start_time TEXT,
                end_time TEXT,
                status TEXT,
                overall_warning_level TEXT,
                current_mean_depth_cm REAL,
                water_area_m2 REAL,
                water_volume_m3 REAL,
                rainfall_intensity_mm_h REAL,
                weather_correction_factor REAL,
                mvp_note TEXT
            );

            CREATE TABLE IF NOT EXISTS stage_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                stage_name TEXT,
                command TEXT,
                config_path TEXT,
                start_time TEXT,
                end_time TEXT,
                status TEXT,
                return_code INTEGER,
                stdout_tail TEXT,
                stderr_tail TEXT
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                artifact_type TEXT,
                path TEXT,
                exists_flag INTEGER,
                note TEXT
            );
            """
        )
        existing_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()
        }
        optional_columns = {
            "final_forecast_5min_cm": "REAL",
            "final_forecast_15min_cm": "REAL",
            "final_forecast_30min_cm": "REAL",
            "final_forecast_60min_cm": "REAL",
            "physical_confidence_summary": "TEXT",
            "forecast_source": "TEXT",
            "s7_pipeline_used": "TEXT",
        }
        for column_name, column_type in optional_columns.items():
            if column_name not in existing_columns:
                conn.execute(f"ALTER TABLE pipeline_runs ADD COLUMN {column_name} {column_type}")


def insert_pipeline_run_start(
    db_path: str | Path,
    run_id: str,
    agent_name: str,
    mode: str,
    start_time: str,
    mvp_note: str,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO pipeline_runs (
                run_id, agent_name, mode, start_time, status, mvp_note
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, agent_name, mode, start_time, "running", mvp_note),
        )


def update_pipeline_run_end(
    db_path: str | Path,
    run_id: str,
    end_time: str,
    status: str,
    overall_warning_level: str | None = None,
    current_mean_depth_cm: float | None = None,
    water_area_m2: float | None = None,
    water_volume_m3: float | None = None,
    rainfall_intensity_mm_h: float | None = None,
    weather_correction_factor: float | None = None,
    final_forecast_5min_cm: float | None = None,
    final_forecast_15min_cm: float | None = None,
    final_forecast_30min_cm: float | None = None,
    final_forecast_60min_cm: float | None = None,
    physical_confidence_summary: str | None = None,
    forecast_source: str | None = None,
    s7_pipeline_used: str | None = None,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE pipeline_runs
            SET end_time = ?,
                status = ?,
                overall_warning_level = ?,
                current_mean_depth_cm = ?,
                water_area_m2 = ?,
                water_volume_m3 = ?,
                rainfall_intensity_mm_h = ?,
                weather_correction_factor = ?,
                final_forecast_5min_cm = ?,
                final_forecast_15min_cm = ?,
                final_forecast_30min_cm = ?,
                final_forecast_60min_cm = ?,
                physical_confidence_summary = ?,
                forecast_source = ?,
                s7_pipeline_used = ?
            WHERE run_id = ?
            """,
            (
                end_time,
                status,
                overall_warning_level,
                current_mean_depth_cm,
                water_area_m2,
                water_volume_m3,
                rainfall_intensity_mm_h,
                weather_correction_factor,
                final_forecast_5min_cm,
                final_forecast_15min_cm,
                final_forecast_30min_cm,
                final_forecast_60min_cm,
                physical_confidence_summary,
                forecast_source,
                s7_pipeline_used,
                run_id,
            ),
        )


def insert_stage_run(
    db_path: str | Path,
    run_id: str,
    stage_name: str,
    command: str,
    config_path: str,
    start_time: str,
    end_time: str,
    status: str,
    return_code: int,
    stdout_tail: str,
    stderr_tail: str,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO stage_runs (
                run_id, stage_name, command, config_path, start_time, end_time,
                status, return_code, stdout_tail, stderr_tail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                stage_name,
                command,
                config_path,
                start_time,
                end_time,
                status,
                return_code,
                stdout_tail,
                stderr_tail,
            ),
        )


def insert_artifact(
    db_path: str | Path,
    run_id: str,
    artifact_type: str,
    path: str,
    exists_flag: int,
    note: str,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO artifacts (
                run_id, artifact_type, path, exists_flag, note
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, artifact_type, path, int(exists_flag), note),
        )


def load_recent_runs(db_path: str | Path, limit: int = 10) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM pipeline_runs
            ORDER BY start_time DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def load_stage_runs(db_path: str | Path, run_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM stage_runs
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def load_artifacts(db_path: str | Path, run_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM artifacts
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
    return [dict(row) for row in rows]
