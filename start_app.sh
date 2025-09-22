#!/bin/bash
PROJECT_DIR="/your_path"

# --- Backend ---
cd "$PROJECT_DIR/backend"

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# --- Frontend ---
cd "$PROJECT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

sleep 5

xdg-open http://localhost:3000

wait $BACKEND_PID $FRONTEND_PID


# --- TODO ---
# chmod +x ~/start_app.sh

# [Desktop Entry]
# Version=1.0
# Type=Application
# Name=Video-Search-Similarity
# Comment=Launch backend + frontend
# Exec=/your_path/start_app.sh
# Icon=utilities-terminal
# Terminal=true
