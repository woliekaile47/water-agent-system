#!/usr/bin/env python3
"""Independent SQLite audit store for non-authoritative C9 shadow records."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def _connect(path: str | Path) -> sqlite3.Connection:
    database = Path(path).expanduser()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    return connection


def init_shadow_db(path: str | Path) -> None:
    with _connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS shadow_runs (
                run_id TEXT PRIMARY KEY,
                protocol_version TEXT NOT NULL,
                monitor_status TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                authoritative INTEGER NOT NULL CHECK(authoritative = 0),
                eligible_for_downstream INTEGER NOT NULL CHECK(eligible_for_downstream = 0),
                formal_warning_generated INTEGER NOT NULL CHECK(formal_warning_generated = 0)
            );
            CREATE TABLE IF NOT EXISTS shadow_sample_states (
                run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                candidate_status TEXT NOT NULL,
                global_estimate_status TEXT NOT NULL,
                result_semantics TEXT NOT NULL,
                area_volume_semantics TEXT NOT NULL,
                mean_depth_cm REAL,
                max_depth_cm REAL,
                water_area_m2 REAL,
                water_volume_m3 REAL,
                s5_status TEXT NOT NULL,
                s7_status TEXT NOT NULL,
                s8_status TEXT NOT NULL CHECK(s8_status = 'warning_suppressed'),
                authoritative INTEGER NOT NULL CHECK(authoritative = 0),
                eligible_for_downstream INTEGER NOT NULL CHECK(eligible_for_downstream = 0),
                PRIMARY KEY (run_id, sample_id)
            );
            """
        )


def write_shadow_audit(
    path: str | Path,
    run_id: str,
    protocol_version: str,
    monitor: dict[str, Any],
    envelopes: list[dict[str, Any]],
) -> None:
    if monitor.get("authoritative") is not False or monitor.get("eligible_for_downstream") is not False:
        raise ValueError("Shadow monitor cannot be authoritative or downstream-eligible")
    init_shadow_db(path)
    with _connect(path) as connection:
        connection.execute(
            """INSERT INTO shadow_runs VALUES (?, ?, ?, ?, 0, 0, 0)""",
            (run_id, protocol_version, monitor["monitor_status"], len(envelopes)),
        )
        for envelope in envelopes:
            state = envelope["canonical_state"]
            values = state["measurements"]
            connection.execute(
                """
                INSERT INTO shadow_sample_states VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0
                )
                """,
                (
                    run_id,
                    state["identity"]["sample_id"],
                    state["identity"]["case_id"],
                    int(state["identity"]["frame_index"]),
                    state["quality"]["candidate_gate"]["status"],
                    state["global_estimate_status"],
                    state["result_semantics"],
                    state["area_volume_semantics"],
                    values["mean_depth_cm"], values["max_depth_cm"],
                    values["water_area_m2"], values["water_volume_m3"],
                    envelope["s5_shadow_input"]["status"],
                    envelope["s7_shadow_preflight"]["status"],
                    envelope["s8_shadow_decision"]["status"],
                ),
            )


def read_shadow_audit(path: str | Path, run_id: str) -> dict[str, Any]:
    with _connect(path) as connection:
        run = connection.execute("SELECT * FROM shadow_runs WHERE run_id = ?", (run_id,)).fetchone()
        samples = connection.execute(
            "SELECT * FROM shadow_sample_states WHERE run_id = ? ORDER BY sample_id", (run_id,)
        ).fetchall()
    if run is None:
        raise KeyError(f"Unknown shadow run_id: {run_id}")
    return {"run": dict(run), "samples": [dict(row) for row in samples]}
