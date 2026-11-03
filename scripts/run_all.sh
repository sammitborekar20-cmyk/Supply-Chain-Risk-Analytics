#!/bin/bash
set -e
python run_pipeline.py "${1:-20000}"
echo "Starting Streamlit dashboard..."
streamlit run app.py
