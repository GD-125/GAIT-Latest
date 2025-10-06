# File: tests/test_dashboard.py
# Dashboard tests

import pytest
from src.dashboard.main_dashboard import MainDashboard

def test_dashboard_initialization():
    dashboard = MainDashboard()
    assert dashboard is not None