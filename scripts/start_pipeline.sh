#!/usr/bin/env bash
# start_pipeline.sh — start the full NIDS-FL Kafka inference pipeline
#
# Services launched:
#   1. FastAPI (api/main.py)         — REST + WebSocket + /inject → Kafka
#   2. Edge Detector (hybrid/edge_detector.py) — binary MLP, consumes network-events
#   3. Cloud Analyzer (hybrid/cloud_analyzer.py) — XGBoost, consumes suspicious-traffic
#
# Prerequisites:
#   docker compose -f infra/docker-compose.yml up -d   (Kafka must be running)
#   source venv/bin/activate
#
# Usage:
#   ./scripts/start_pipeline.sh
#   ./scripts/start_pipeline.sh --threshold 0.90 --batch-size 64

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="./venv/bin/python"
LOG_DIR="logs/pipeline"
BROKER="localhost:29092"
THRESHOLD="0.95"
BATCH_SIZE="32"
API_HOST="0.0.0.0"
API_PORT="8000"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threshold)  THRESHOLD="$2";  shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --broker)     BROKER="$2";     shift 2 ;;
    --port)       API_PORT="$2";   shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"

echo ""
echo "  NIDS-FL Kafka Pipeline"
echo "  ─────────────────────────────────────────────────"
echo "  Broker    : $BROKER"
echo "  Threshold : $THRESHOLD   Batch : $BATCH_SIZE"
echo "  API       : http://$API_HOST:$API_PORT"
echo "  Logs      : $LOG_DIR/"
echo "  ─────────────────────────────────────────────────"
echo ""

# ── Verify Kafka is reachable ─────────────────────────────────────────────────
echo "[pipeline] Checking Kafka broker at $BROKER …"
if ! $PYTHON -c "
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
try:
    c = KafkaConsumer(bootstrap_servers=['$BROKER'], consumer_timeout_ms=3000)
    c.close()
    print('  Kafka OK')
except NoBrokersAvailable:
    print('  ERROR: Kafka not reachable')
    exit(1)
"; then
  echo ""
  echo "  Start Kafka first:"
  echo "    docker compose -f infra/docker-compose.yml up -d"
  exit 1
fi

# ── Kill any stale pipeline processes ─────────────────────────────────────────
pkill -f "uvicorn api.main"      2>/dev/null && echo "[pipeline] Stopped old API" || true
pkill -f "edge_detector"         2>/dev/null && echo "[pipeline] Stopped old edge_detector" || true
pkill -f "cloud_analyzer"        2>/dev/null && echo "[pipeline] Stopped old cloud_analyzer" || true
sleep 1

# ── 1. FastAPI ─────────────────────────────────────────────────────────────────
echo "[pipeline] Starting API on port $API_PORT …"
KAFKA_BROKER="$BROKER" uvicorn api.main:app \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --log-level info \
  > "$LOG_DIR/api.log" 2>&1 &
API_PID=$!
echo "[pipeline] API PID=$API_PID → $LOG_DIR/api.log"

# Wait for API to be ready before starting consumers
echo "[pipeline] Waiting for API to be ready …"
for i in $(seq 1 20); do
  if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
    echo "[pipeline] API ready (${i}s)"
    break
  fi
  sleep 1
done

# ── 2. Edge Detector ──────────────────────────────────────────────────────────
echo "[pipeline] Starting Edge Detector (threshold=$THRESHOLD, batch=$BATCH_SIZE) …"
KAFKA_BROKER="$BROKER" $PYTHON -m hybrid.edge_detector \
  --broker     "$BROKER" \
  --threshold  "$THRESHOLD" \
  --batch-size "$BATCH_SIZE" \
  > "$LOG_DIR/edge_detector.log" 2>&1 &
EDGE_PID=$!
echo "[pipeline] Edge Detector PID=$EDGE_PID → $LOG_DIR/edge_detector.log"

# Give model time to load before cloud analyzer starts
sleep 3

# ── 3. Cloud Analyzer ─────────────────────────────────────────────────────────
echo "[pipeline] Starting Cloud Analyzer …"
KAFKA_BROKER="$BROKER" $PYTHON -m hybrid.cloud_analyzer \
  --broker "$BROKER" \
  > "$LOG_DIR/cloud_analyzer.log" 2>&1 &
CLOUD_PID=$!
echo "[pipeline] Cloud Analyzer PID=$CLOUD_PID → $LOG_DIR/cloud_analyzer.log"

# ── Save PIDs ─────────────────────────────────────────────────────────────────
echo "$API_PID"   > "$LOG_DIR/pids"
echo "$EDGE_PID"  >> "$LOG_DIR/pids"
echo "$CLOUD_PID" >> "$LOG_DIR/pids"

echo ""
echo "  Pipeline running. Data flow:"
echo "  traffic_gen --direct → POST /inject → network-events"
echo "  → edge_detector → suspicious-traffic"
echo "  → cloud_analyzer → classified-attacks / alerts"
echo "  → API WebSocket → Dashboard"
echo ""
echo "  Stop with:  ./scripts/stop_pipeline.sh"
echo "  Test with:  python scripts/traffic_gen.py --direct --attack-type mixed --duration 60"
echo ""
echo "[pipeline] Tailing edge_detector log (Ctrl+C to detach, processes keep running):"
echo ""
tail -f "$LOG_DIR/edge_detector.log"
