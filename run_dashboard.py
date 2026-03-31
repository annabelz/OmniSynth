"""
Thin launcher for the Streamlit dashboard.

Usage:
    streamlit run run_dashboard.py
    streamlit run run_dashboard.py -- --config configs/my_config.yaml
"""
from stdg_eval.visualization.dashboard import run_dashboard

run_dashboard()
