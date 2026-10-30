#!/usr/bin/env bash
# stop_all.sh — stop all FL training processes
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="logs/fl"
PIDS_FILE="$LOG_DIR/pids"

echo "[FL] Stopping all FL processes..."

# Kill by saved PIDs first
if [[ -f "$PIDS_FILE" ]]; then
  while IFS= read -r pid; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "[FL] Killed PID $pid"
    fi
  done < "$PIDS_FILE"
  rm -f "$PIDS_FILE"
fi

# Fallback: kill by process name (catches any strays)
pkill -f "fl_server" 2>/dev/null && echo "[FL] Killed fl_server processes" || true
pkill -f "fl_client" 2>/dev/null && echo "[FL] Killed fl_client processes" || true

echo "[FL] Done."
