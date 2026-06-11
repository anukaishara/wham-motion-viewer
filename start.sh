#!/usr/bin/env bash
# =============================================================================
# WHAM Unified App — start script
#
# Starts both the FastAPI backend and the Vite dev frontend.
# Open http://localhost:5173 in your browser.
#
# Usage:
#   chmod +x start.sh
#   ./start.sh
#
# For production (static build):
#   cd /home/sensor_readings/IPCV_Prj/motion_viewer && npm run build
#   ./start.sh --prod         # backend only, serves React from /dist at :8787
# =============================================================================

set -e

WHAM_DIR="$(cd "$(dirname "$0")" && pwd)"
VIEWER_DIR="$WHAM_DIR/motion_viewer"
PYTHON="$HOME/miniconda3/envs/wham_dev/bin/python"

if [[ "$1" == "--prod" ]]; then
    echo "Starting WHAM server in production mode → http://localhost:8787"
    cd "$WHAM_DIR"
    "$PYTHON" server.py
    exit 0
fi

# Development mode: backend on :8787, Vite dev on :5173
echo "Starting WHAM backend  → http://localhost:8787/api/"
echo "Starting Vite frontend → http://localhost:5173"
echo "Open http://localhost:5173 in your browser"
echo ""

# Start backend in background
cd "$WHAM_DIR"
"$PYTHON" server.py &
BACKEND_PID=$!

# Start Vite dev server
cd "$VIEWER_DIR"
npm run dev &
FRONTEND_PID=$!

# Trap Ctrl+C and kill both
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

wait
