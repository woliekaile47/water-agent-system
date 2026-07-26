#!/usr/bin/env python3
"""Dedicated Streamlit entry point for the simulation-road competition demo."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_DASHBOARD_DIR = Path(__file__).resolve().parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from app import page_competition_simulation_demo


def main() -> None:
    st.set_page_config(
        page_title="预鉴：仿真道路积水闭环演示",
        page_icon="W",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    page_competition_simulation_demo()


if __name__ == "__main__":
    main()
