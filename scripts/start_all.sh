#!/usr/bin/env bash
# start_all.sh — start the full NIDS-FL stack
# Usage:
#   ./scripts/start_all.sh              # one-shot 10 rounds
#   ./scripts/start_all.sh --daily      # repeat every 24 h
#   ./scripts/start_all.sh --weekly     # repeat every 168 h
#   ./scripts/start_all.sh --hours 48   # repeat every 48 h

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="./venv/bin/python"
LOG_DIR="logs/fl"
REPEAT_HOURS=0
ROUNDS=10
EPOCHS=5
CLIENTS=3
BROKER="localhost:29092"
TIMEOUT=120

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --daily)   REPEAT_HOURS=24;  shift ;;
    --weekly)  REPEAT_HOURS=168; shift ;;
    --hours)   REPEAT_HOURS="$2"; shift 2 ;;
    --rounds)  ROUNDS="$2";  shift 2 ;;
    --epochs)  EPOCHS="$2";  shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

REPEAT_FLAG=""
if (( $(echo "$REPEAT_HOURS > 0" | bc -l) )); then
  REPEAT_FLAG="--repeat-hours $REPEAT_HOURS"
  echo "[FL] Continuous mode — repeating every ${REPEAT_HOURS}h"
else
  echo "[FL] One-shot mode — $ROUNDS rounds"
fi

# ── Kill any leftover FL processes ────────────────────────────────────────────
pkill -f "fl_server" 2>/dev/null || true
pkill -f "fl_client" 2>/dev/null || true
sleep 1

# ── Launch server ─────────────────────────────────────────────────────────────
echo "[FL] Starting server..."
$PYTHON -m federated_learning.fl_server \
  --rounds  "$ROUNDS" \
  --clients "$CLIENTS" \
  --broker  "$BROKER" \
  --timeout "$TIMEOUT" \
  $REPEAT_FLAG \
  > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "[FL] Server PID=$SERVER_PID → $LOG_DIR/server.log"

# Give server a moment to initialize Kafka consumer before clients connect
sleep 3

# ── Launch clients ────────────────────────────────────────────────────────────
CLIENT_PIDS=()
for ID in 1 2 3; do
  echo "[FL] Starting client $ID..."
  $PYTHON -m federated_learning.fl_client \
    --id     "$ID" \
    --rounds "$ROUNDS" \
    --epochs "$EPOCHS" \
    --broker "$BROKER" \
    $REPEAT_FLAG \
    > "$LOG_DIR/client_${ID}.log" 2>&1 &
  CLIENT_PIDS+=($!)
  echo "[FL] Client $ID PID=${CLIENT_PIDS[-1]} → $LOG_DIR/client_${ID}.log"
done

# ── Save PIDs ─────────────────────────────────────────────────────────────────
echo "$SERVER_PID"           > "$LOG_DIR/pids"
printf '%s\n' "${CLIENT_PIDS[@]}" >> "$LOG_DIR/pids"

echo ""
echo "[FL] All processes started. Tailing server log (Ctrl+C to detach, processes keep running):"
echo "     Stop everything with: ./scripts/stop_all.sh"
echo ""
tail -f "$LOG_DIR/server.log"
