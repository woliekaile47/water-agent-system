#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root() / path


def read_json(path_value: str | Path) -> dict[str, Any] | list[Any] | None:
    path = resolve_path(path_value)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path_value: str | Path) -> str | None:
    path = resolve_path(path_value)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def show_missing(path_value: str | Path) -> None:
    st.warning(f"缺少文件：`{path_value}`")


def show_image(path_value: str | Path, caption: str | None = None) -> bool:
    path = resolve_path(path_value)
    if not path.exists():
        show_missing(path_value)
        return False
    st.image(str(path), caption=caption or str(path_value), use_container_width=True)
    return True


def show_json(path_value: str | Path, title: str | None = None) -> bool:
    if title:
        st.subheader(title)
    data = read_json(path_value)
    if data is None:
        show_missing(path_value)
        return False
    st.json(data, expanded=False)
    return True


def show_markdown(path_value: str | Path, title: str | None = None) -> bool:
    if title:
        st.subheader(title)
    text = read_text(path_value)
    if text is None:
        show_missing(path_value)
        return False
    st.markdown(text)
    return True


def metric_value(data: dict[str, Any] | None, key: str, default: str = "N/A") -> Any:
    if not data:
        return default
    value = data.get(key)
    if value is None:
        return default
    return value


def format_number(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None or value == "N/A":
        return "N/A"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def metric_grid(items: list[tuple[str, Any, str]]) -> None:
    if not items:
        return
    columns = st.columns(min(4, len(items)))
    for index, (label, value, help_text) in enumerate(items):
        with columns[index % len(columns)]:
            st.metric(label, value, help=help_text or None)


def json_table(data: dict[str, Any] | None, keys: list[str]) -> None:
    if not data:
        st.info("暂无可展示的结构化指标。")
        return
    rows = [{"指标": key, "值": data.get(key, "N/A")} for key in keys]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def section_note(text: str) -> None:
    st.info(text)


def status_badge(label: str, value: Any) -> None:
    if str(value).lower() in {"reject", "false", "none"}:
        st.error(f"{label}: {value}")
    elif str(value).lower() in {"warning", "low_risk", "suspected_water"}:
        st.warning(f"{label}: {value}")
    else:
        st.success(f"{label}: {value}")


def file_summary(paths: list[str]) -> None:
    rows = []
    for path_value in paths:
        path = resolve_path(path_value)
        rows.append({"文件": path_value, "状态": "存在" if path.exists() else "缺失"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
