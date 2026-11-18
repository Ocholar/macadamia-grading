#!/bin/bash
# docker-entrypoint.sh
# Launches FastAPI backend and Streamlit dashboard simultaneously.

set -e

echo "============================================="
echo "  Macadamia Quality Grading AI System v1.0"
echo "============================================="
echo "  API:  http://localhost:8000"
echo "  UI:   http://localhost:8501"
echo "  Docs: http://localhost:8000/docs"
echo "============================================="

# Start FastAPI backend in the background
python -m uvicorn deployment.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info &

API_PID=$!
echo "[API] FastAPI started (PID $API_PID)"

# Start Streamlit dashboard in the foreground
streamlit run deployment/ui/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.base dark \
    --theme.primaryColor "#58A6FF" \
    --theme.backgroundColor "#0F1117" \
    --theme.secondaryBackgroundColor "#161B22" \
    --theme.textColor "#E8ECF0"
