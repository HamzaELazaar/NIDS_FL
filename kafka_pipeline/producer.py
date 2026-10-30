"""WebSocket → Kafka bridge for the NIDS-FL pipeline.

Consumes live packet features from the FastAPI WebSocket endpoint and
produces them into the 'network-events' Kafka topic.

Run:
    python kafka_pipeline/producer.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import websockets
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

# Allow importing api.features from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "localhost:29092")
TOPIC: str = os.getenv("TOPIC_NETWORK_EVENTS", "network-events")
_raw_host: str = os.getenv("API_HOST", "0.0.0.0")
API_HOST: str = "localhost" if _raw_host in ("0.0.0.0", "") else _raw_host
API_PORT: int = int(os.getenv("API_PORT", "8000"))
WS_URL: str = f"ws://{API_HOST}:{API_PORT}/stream"

# Reconnection backoff: 1s → 2s → 4s → … → 60s max
_BACKOFF_BASE: float = 1.0
_BACKOFF_MAX: float = 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Runtime stats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Stats:
    sent: int = 0
    errors: int = 0
    _window_sent: int = 0
    _window_start: float = field(default_factory=time.time)

    def record_sent(self) -> None:
        self.sent += 1
        self._window_sent += 1

    def record_error(self) -> None:
        self.errors += 1

    def flush(self) -> dict:
        """Return stats for the last window and reset the counter."""
        elapsed = time.time() - self._window_start or 1.0
        mps = round(self._window_sent / elapsed, 2)
        snapshot = {
            "total_sent": self.sent,
            "msg_per_sec": mps,
            "errors": self.errors,
        }
        self._window_sent = 0
        self._window_start = time.time()
        return snapshot


_stats = _Stats()


# ─────────────────────────────────────────────────────────────────────────────
# Kafka producer factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_producer() -> KafkaProducer:
    """Create and return a KafkaProducer with JSON value serializer.

    Raises:
        NoBrokersAvailable: if Kafka is not reachable.
    """
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k if isinstance(k, bytes) else k.encode("utf-8"),
        acks="all",
        retries=3,
        request_timeout_ms=5_000,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stats reporter (background task)
# ─────────────────────────────────────────────────────────────────────────────

async def _report_stats(interval: float = 10.0) -> None:
    """Log runtime statistics every *interval* seconds."""
    while True:
        await asyncio.sleep(interval)
        s = _stats.flush()
        logger.info(
            "📊 Stats — sent: %d  msg/s: %.2f  errors: %d",
            s["total_sent"], s["msg_per_sec"], s["errors"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core bridge
# ─────────────────────────────────────────────────────────────────────────────

async def stream_to_kafka(producer: KafkaProducer) -> None:
    """Connect to the FastAPI WebSocket and forward every message to Kafka.

    Reconnects automatically with exponential backoff if the WebSocket drops.

    Args:
        producer: An initialised KafkaProducer instance.
    """
    backoff = _BACKOFF_BASE

    while True:
        try:
            logger.info("🔌 Connecting to %s …", WS_URL)
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=10) as ws:
                logger.info("✅ WebSocket connected")
                backoff = _BACKOFF_BASE  # reset on successful connection

                async for raw in ws:
                    try:
                        msg: dict = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        logger.warning("⚠️  Bad JSON from WS: %s", exc)
                        continue

                    # The API /stream is shared by the Kafka bridge and the
                    # dashboard. Only API-captured packet events should be
                    # published to Kafka; Kafka-origin dashboard events would
                    # otherwise loop back into the same topics.
                    if msg.get("source") == "kafka":
                        continue
                    if msg.get("type") and msg.get("type") != "packet":
                        continue

                    # Extract the raw features and convert to the 11 CICIDS2017
                    # keys that edge_detector expects.
                    raw_features: dict = msg.get("data") or msg.get("features") or msg

                    try:
                        from api.features import to_ml_vector
                        ml_vec = to_ml_vector(raw_features)
                    except Exception:
                        # Fallback: if already CICIDS-keyed, use as-is
                        ml_vec = {k: v for k, v in raw_features.items() if isinstance(v, (int, float))}

                    # Attach routing context (src/dst) for downstream alerts
                    ml_vec["src_ip"]   = raw_features.get("src_ip",   "")
                    ml_vec["dst_ip"]   = raw_features.get("dst_ip",   "")
                    ml_vec["src_port"] = raw_features.get("src_port", 0)
                    ml_vec["dst_port"] = raw_features.get("dst_port", 0)
                    ml_vec["protocol"] = raw_features.get("protocol", "?")
                    ml_vec["timestamp"] = raw_features.get("timestamp", time.time())

                    src_ip:   str = ml_vec["src_ip"]
                    protocol: str = ml_vec["protocol"]
                    src_port: int = ml_vec["src_port"]
                    dst_ip:   str = ml_vec["dst_ip"]
                    dst_port: int = ml_vec["dst_port"]

                    try:
                        future = producer.send(
                            TOPIC,
                            key=src_ip or "unknown",
                            value=ml_vec,
                        )
                        # Resolve the future off the event loop to avoid blocking it
                        await asyncio.to_thread(future.get, 5)
                        _stats.record_sent()
                        logger.info(
                            "📨 %s  %s:%d → %s:%d",
                            protocol, src_ip, src_port, dst_ip, dst_port,
                        )
                    except KafkaError as exc:
                        _stats.record_error()
                        logger.error("❌ Kafka send failed: %s", exc)

        except (websockets.ConnectionClosed, websockets.InvalidURI) as exc:
            logger.warning("🔁 WS closed (%s) — retry in %.0fs", exc, backoff)
        except OSError as exc:
            logger.warning("🔁 WS unreachable (%s) — retry in %.0fs", exc, backoff)

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, _BACKOFF_MAX)


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Bootstrap the bridge: connect to Kafka, then start the async loop."""
    logger.info("🚀 NIDS-FL Kafka producer starting")
    logger.info("   Kafka  : %s → topic '%s'", KAFKA_BROKER, TOPIC)
    logger.info("   API WS : %s", WS_URL)

    # Retry Kafka connection at startup
    producer: KafkaProducer | None = None
    for attempt in range(1, 6):
        try:
            producer = _make_producer()
            logger.info("✅ Kafka connected on attempt %d", attempt)
            break
        except NoBrokersAvailable:
            wait = min(2 ** attempt, 30)
            logger.warning(
                "⚠️  Kafka not reachable (attempt %d/5) — retry in %ds\n"
                "   Check: docker compose -f infra/docker-compose.yml ps",
                attempt, wait,
            )
            time.sleep(wait)

    if producer is None:
        logger.error("❌ Could not connect to Kafka after 5 attempts — aborting")
        raise SystemExit(1)

    async def _run() -> None:
        await asyncio.gather(
            stream_to_kafka(producer),
            _report_stats(10.0),
        )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("⛔ Interrupted — flushing Kafka producer …")
    finally:
        if producer:
            producer.flush()
            producer.close()
            logger.info("👋 Producer closed cleanly")


if __name__ == "__main__":
    main()
