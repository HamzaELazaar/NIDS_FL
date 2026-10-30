#!/usr/bin/env bash
# stop_pipeline.sh — stop all NIDS-FL Kafka pipeline processes
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="logs/pipeline"
PIDS_FILE="$LOG_DIR/pids"

echo "[pipeline] Stopping all pipeline processes..."

# Kill by saved PIDs first
if [[ -f "$PIDS_FILE" ]]; then
  while IFS= read -r pid; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "[pipeline] Killed PID $pid"
    fi
  done < "$PIDS_FILE"
  rm -f "$PIDS_FILE"
fi

# Fallback: kill by process name
pkill -f "uvicorn api.main"  2>/dev/null && echo "[pipeline] Killed API" || true
pkill -f "edge_detector"     2>/dev/null && echo "[pipeline] Killed edge_detector" || true
pkill -f "cloud_analyzer"    2>/dev/null && echo "[pipeline] Killed cloud_analyzer" || true

echo "[pipeline] Done."
