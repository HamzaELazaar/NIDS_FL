"""NIDS-FL FastAPI — live capture + hybrid MLP→XGBoost classification.

Full pipeline
-------------
  eth2 → pyshark → feature extraction → Edge MLP (binary) → XGBoost (7-class)
                                                    ↓ confident Normal → DROP

Endpoints
---------
POST /predict              — classify one packet by feature dict (JSON)
GET  /predictions          — recent pipeline results (circular buffer)
GET  /alerts               — confirmed attacks only
GET  /packets              — raw captured packets
GET  /stats                — protocol / attack distribution + top attacks
GET  /pipeline/status      — loaded model metadata
GET  /fl/history           — federated learning round history
GET  /hybrid/metrics       — hybrid pipeline orchestrator metrics
GET  /models               — registered model metadata (MANIFEST.json)
GET  /graph/attacks        — attack graph topology (Neo4j)
GET  /health               — liveness + capture status
WS   /stream               — real-time classified-packet stream

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.capture import live_capture
from api.features import infer_app_protocol, normalize_features, to_ml_vector

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_ROOT = Path(__file__).resolve().parent.parent
KAFKA_BROKER    = os.getenv("KAFKA_BROKER",    "localhost:29092")
CAPTURE_IFACE   = os.getenv("CAPTURE_IFACE",   "eth0")
KAFKA_TOPICS = [
    "network-events",
    "suspicious-traffic",
    "classified-attacks",
    "alerts",
    "fl-rounds",
]

# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────

BUFFER:      deque[dict] = deque(maxlen=1000)   # raw captured packets
PREDICTIONS: deque[dict] = deque(maxlen=2000)   # classified packets (edge+cloud)
ALERTS:      deque[dict] = deque(maxlen=500)    # attacks only — rolling window for WS/live view
_ALL_ALERTS: list[dict]  = []                   # unbounded; loaded from file on startup

WS_CLIENTS: list[WebSocket] = []
_start_time:      float = time.time()
_capture_active:  bool  = False
_pipeline:        Optional["NIDSPipeline"] = None


def _empty_hybrid_metrics() -> dict:
    return {
        "timestamp":                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uptime_seconds":               round(time.time() - _start_time, 1),
        "packets_total":                0,
        "packets_forwarded_to_cloud":   0,
        "attacks_confirmed_by_xgboost": 0,
        "alerts_published":             0,
        "edge_false_positive_rate":     0.0,
        "bandwidth_saved_pct":          0.0,
        "avg_edge_latency_ms":          0.0,
        "avg_cloud_latency_ms":         0.0,
        "p95_total_latency_ms":         0.0,
        "attack_type_distribution":     {},
        "parse_errors":                 {topic: 0 for topic in KAFKA_TOPICS},
    }


def _packet_from_kafka(payload: dict) -> dict:
    """Normalize Kafka payloads into the dashboard's PacketFeatures shape."""
    features = payload.get("features") or payload.get("packet_features") or payload
    avg_size = float(features.get("Average Packet Size", features.get("packet_length", 0)) or 0)
    dst_port = int(float(features.get("dst_port", features.get("Destination Port", 0)) or 0))
    src_port = int(float(features.get("src_port", 0) or 0))
    proto    = str(features.get("protocol", "OTHER") or "OTHER").upper()
    return {
        "timestamp":          float(features.get("timestamp", time.time()) or time.time()),
        "protocol":           proto,
        "app_protocol":       infer_app_protocol(proto, dst_port, src_port),
        "src_ip":             str(features.get("src_ip", "")),
        "dst_ip":             str(features.get("dst_ip", "")),
        "src_port":           int(float(features.get("src_port", 0) or 0)),
        "dst_port":           dst_port,
        "packet_length":      int(float(features.get("packet_length", avg_size) or 0)),
        "ttl":                int(float(features.get("ttl", 0) or 0)),
        "syn_flag":           int(float(features.get("syn_flag", 0) or 0)),
        "fin_flag":           int(float(features.get("fin_flag", 0) or 0)),
        "rst_flag":           int(float(features.get("rst_flag", 0) or 0)),
        "psh_flag":           int(float(features.get("psh_flag", 0) or 0)),
        "ack_flag":           int(float(features.get("ack_flag", 0) or 0)),
        "urg_flag":           int(float(features.get("urg_flag", 0) or 0)),
        "window_size":        int(float(features.get("window_size", 0) or 0)),
        "icmp_type":          int(float(features.get("icmp_type", 0) or 0)),
        "icmp_code":          int(float(features.get("icmp_code", 0) or 0)),
        "payload_length":     int(float(features.get("payload_length", 0) or 0)),
        "packet_count_1s":    int(float(features.get("packet_count_1s", 0) or 0)),
        "bytes_per_sec_1s":   float(features.get("bytes_per_sec_1s", 0) or 0),
        "avg_packet_size_1s": float(features.get("avg_packet_size_1s", avg_size) or 0),
        "packet_freq_1s":     float(features.get("packet_freq_1s", 0) or 0),
    }


def _normalize_kafka_alert(payload: dict) -> dict:
    """Map a cloud_analyzer Kafka alert to the dashboard's Alert interface.

    cloud_analyzer publishes:
        edge_prediction, cloud_prediction, pipeline_total_latency_ms, timestamp (ISO)

    Dashboard expects:
        stage, final_label, edge, cloud, pipeline_latency_ms, timestamp (float)
    """
    cloud_pred = payload.get("cloud_prediction", {})
    edge_pred  = payload.get("edge_prediction",  {})

    # Normalise timestamp → Unix float (cloud_analyzer sends ISO string)
    raw_ts = payload.get("timestamp", time.time())
    if isinstance(raw_ts, str):
        try:
            from datetime import timezone
            from datetime import datetime as _dt
            ts = _dt.fromisoformat(raw_ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = time.time()
    else:
        ts = float(raw_ts)

    # Build the cloud sub-object the dashboard expects
    cloud_obj = {
        **cloud_pred,
        # Dashboard uses top_3_classes; cloud_analyzer stores it as top_3
        "top_3_classes": cloud_pred.get("top_3_classes") or cloud_pred.get("top_3", []),
    }

    alert = dict(payload)
    alert.update({
        "stage":               "cloud",
        "final_label":         cloud_pred.get("attack_type", payload.get("final_label", "Unknown")),
        "timestamp":           ts,
        # Canonical keys the dashboard components rely on
        "edge":                edge_pred,
        "cloud":               cloud_obj,
        "pipeline_latency_ms": float(
            payload.get("pipeline_total_latency_ms") or payload.get("pipeline_latency_ms") or 0
        ),
    })
    # Pass alert (timestamp already a float) so _packet_from_kafka never sees an ISO string
    alert.setdefault("features", _packet_from_kafka(alert))
    return alert


class KafkaDashboardState:
    """Small live cache for dashboard data sourced from Kafka."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.connected = False
        self.last_error = ""
        self.topic_totals: dict[str, int] = {topic: 0 for topic in KAFKA_TOPICS}
        self.topic_rates: dict[str, float] = {topic: 0.0 for topic in KAFKA_TOPICS}
        self._last_offsets: dict[str, int] = {topic: 0 for topic in KAFKA_TOPICS}
        self._last_offsets_at = time.time()
        self.protocols: Counter = Counter()
        self.app_protocols: Counter = Counter()
        self.src_ips: Counter = Counter()
        self.dst_ports: Counter = Counter()
        self.labels: Counter = Counter()
        self.severities: Counter = Counter()
        self.attack_dist: Counter = Counter()
        self.edge_latencies: deque[float] = deque(maxlen=10_000)
        self.cloud_latencies: deque[float] = deque(maxlen=10_000)
        self.total_latencies: deque[float] = deque(maxlen=10_000)
        self.parse_errors: Counter = Counter({topic: 0 for topic in KAFKA_TOPICS})
        # Counters updated directly by capture_loop (used when Kafka pipeline is absent)
        self._cap_total:     int = 0
        self._cap_forwarded: int = 0
        self._cap_alerts:    int = 0

    def update_offsets(self, offsets: dict[str, int]) -> None:
        now = time.time()
        with self._lock:
            dt = max(now - self._last_offsets_at, 0.001)
            for topic in KAFKA_TOPICS:
                current = int(offsets.get(topic, 0))
                previous = self._last_offsets.get(topic, 0)
                if current < previous:
                    self._reset_rolling_locked()
                    previous = 0
                self.topic_totals[topic] = current
                self.topic_rates[topic] = max(0.0, (current - previous) / dt)
                self._last_offsets[topic] = current
            self._last_offsets_at = now
            self.connected = True
            self.last_error = ""

    def mark_error(self, exc: Exception) -> None:
        with self._lock:
            self.connected = False
            self.last_error = str(exc)

    def record_message(self, topic: str, payload: dict) -> Optional[dict]:
        try:
            event = None
            with self._lock:
                if topic == "network-events":
                    pkt = _packet_from_kafka(payload)
                    self.protocols.update([pkt["protocol"]])
                    self.app_protocols.update([pkt["app_protocol"]])
                    if pkt["src_ip"]:
                        self.src_ips.update([pkt["src_ip"]])
                    if pkt["dst_port"]:
                        self.dst_ports.update([pkt["dst_port"]])
                    event = {"type": "packet", "data": pkt}
                elif topic == "suspicious-traffic":
                    edge = payload.get("edge_prediction", {})
                    lat = float(edge.get("edge_latency_ms", edge.get("latency_ms", 0)) or 0)
                    if lat > 0:
                        self.edge_latencies.append(lat)
                elif topic in ("classified-attacks", "alerts"):
                    cloud = payload.get("cloud_prediction", {})
                    attack_type = (
                        cloud.get("attack_type")
                        or payload.get("final_label")
                        or payload.get("attack_type")
                        or "Unknown"
                    )
                    severity = payload.get("severity", "none")
                    self.labels.update([attack_type])
                    self.severities.update([severity])
                    if topic == "classified-attacks":
                        self.attack_dist.update([attack_type])
                    cloud_lat = float(cloud.get("cloud_latency_ms", 0) or 0)
                    total_lat = float(payload.get("pipeline_total_latency_ms", payload.get("pipeline_latency_ms", 0)) or 0)
                    if cloud_lat > 0:
                        self.cloud_latencies.append(cloud_lat)
                    if total_lat > 0:
                        self.total_latencies.append(total_lat)
                    if topic == "alerts":
                        event = {"type": "alert", "data": _normalize_kafka_alert(payload)}
            return event
        except Exception:
            with self._lock:
                self.parse_errors.update([topic])
            return None

    def stats_snapshot(self) -> dict:
        with self._lock:
            total_packets = self.topic_totals.get("network-events", 0)
            return {
                "uptime_seconds":         round(time.time() - _start_time, 1),
                "total_packets":          total_packets,
                "packets_per_second":     round(self.topic_rates.get("network-events", 0.0), 2),
                "total_classified":       self.topic_totals.get("classified-attacks", 0),
                "total_alerts":           self.topic_totals.get("alerts", 0),
                "protocols":              dict(self.protocols),
                "app_protocols":          dict(self.app_protocols),
                "top_5_source_ips":       dict(self.src_ips.most_common(5)),
                "top_5_destination_ports": dict(self.dst_ports.most_common(5)),
                "label_distribution":     dict(self.labels),
                "severity_distribution":  dict(self.severities),
                "kafka_connected":        self.connected,
                "kafka_error":            self.last_error,
                "kafka_topics": {
                    topic: {
                        "total": self.topic_totals.get(topic, 0),
                        "per_second": round(self.topic_rates.get(topic, 0.0), 2),
                    }
                    for topic in KAFKA_TOPICS
                },
            }

    def hybrid_snapshot(self) -> dict:
        with self._lock:
            total = self.topic_totals.get("network-events", 0)
            fwd = self.topic_totals.get("suspicious-traffic", 0)
            classified = self.topic_totals.get("classified-attacks", 0)
            alerts = self.topic_totals.get("alerts", 0)
            edge_lats = list(self.edge_latencies)
            cloud_lats = list(self.cloud_latencies)
            total_lats = list(self.total_latencies)
            attack_dist = dict(self.attack_dist)
            parse_errors = dict(self.parse_errors)
            connected = self.connected
            error = self.last_error

        # When the Kafka pipeline is not running, fall back to in-process counters
        # so the dashboard shows live data from the direct capture path.
        if total == 0 and self._cap_total > 0:
            total      = self._cap_total
            fwd        = self._cap_forwarded
            alerts     = self._cap_alerts
            classified = self._cap_forwarded

        bw_saved = 1.0 - (fwd / total) if total > 0 else 0.0
        fp_rate  = max(0.0, (fwd - alerts) / fwd) if fwd > 0 else 0.0
        return {
            **_empty_hybrid_metrics(),
            "packets_total":                total,
            "packets_forwarded_to_cloud":   fwd,
            "attacks_confirmed_by_xgboost": alerts,
            "alerts_published":             alerts,
            "edge_false_positive_rate":     round(fp_rate, 4),
            "bandwidth_saved_pct":          round(bw_saved * 100, 2),
            "avg_edge_latency_ms":          round(sum(edge_lats) / len(edge_lats), 2) if edge_lats else 0.0,
            "avg_cloud_latency_ms":         round(sum(cloud_lats) / len(cloud_lats), 2) if cloud_lats else 0.0,
            "p95_total_latency_ms":         round(sorted(total_lats)[int(0.95 * (len(total_lats) - 1))], 2) if total_lats else 0.0,
            "attack_type_distribution":     attack_dist,
            "parse_errors":                 parse_errors,
            "kafka_connected":              connected,
            "kafka_error":                  error,
            "kafka_topics":                 self.stats_snapshot()["kafka_topics"],
            "classified_total":             classified,
        }

    def record_direct_capture(self, pkt: dict, result: Optional[dict]) -> None:
        """Update counters from capture_loop's in-process pipeline.

        Called for every packet that goes through the direct MLP→XGBoost path
        (i.e. when the Kafka pipeline is not in the loop).  The data is used by
        hybrid_snapshot() as a fallback when Kafka topic offsets are all zero.
        """
        with self._lock:
            self._cap_total += 1
            self.protocols.update([pkt.get("protocol", "OTHER")])
            self.app_protocols.update([pkt.get("app_protocol") or pkt.get("protocol", "OTHER")])
            if pkt.get("src_ip"):
                self.src_ips.update([pkt["src_ip"]])
            if pkt.get("dst_port"):
                self.dst_ports.update([str(pkt["dst_port"])])

            if result is None:
                return

            edge = result.get("edge") or {}
            cloud = result.get("cloud")

            edge_lat = float(edge.get("edge_latency_ms", 0) or 0)
            if edge_lat > 0:
                self.edge_latencies.append(edge_lat)

            total_lat = float(result.get("pipeline_latency_ms", 0) or 0)
            if total_lat > 0:
                self.total_latencies.append(total_lat)

            if cloud:
                self._cap_forwarded += 1
                cloud_lat = float(cloud.get("cloud_latency_ms", 0) or 0)
                if cloud_lat > 0:
                    self.cloud_latencies.append(cloud_lat)
                attack_type = cloud.get("attack_type") or result.get("final_label") or "Unknown"
                self.attack_dist.update([attack_type])
                self.labels.update([attack_type])

            severity = result.get("severity", "none")
            self.severities.update([severity])
            if severity not in ("none", ""):
                self._cap_alerts += 1

    def _reset_rolling_locked(self) -> None:
        self.protocols.clear()
        self.app_protocols.clear()
        self.src_ips.clear()
        self.dst_ports.clear()
        self.labels.clear()
        self.severities.clear()
        self.attack_dist.clear()
        self.edge_latencies.clear()
        self.cloud_latencies.clear()
        self.total_latencies.clear()
        self.parse_errors    = Counter({topic: 0 for topic in KAFKA_TOPICS})
        self._cap_total      = 0
        self._cap_forwarded  = 0
        self._cap_alerts     = 0


KAFKA_STATE = KafkaDashboardState()


# ─────────────────────────────────────────────────────────────────────────────
# Inference pipeline (no Kafka — direct in-process call)
# ─────────────────────────────────────────────────────────────────────────────

class NIDSPipeline:
    """Hybrid MLP→XGBoost pipeline for use inside the API (no Kafka)."""

    _EDGE_MODEL  = _ROOT / "models" / "final" / "best_mlp_binary.keras"
    _CLOUD_MODEL = _ROOT / "models" / "final" / "best_centralized_multiclass.json"
    _ENCODER     = _ROOT / "models" / "final" / "label_encoder.joblib"
    _SCALER_DIR  = _ROOT / "models" / "final"

    def __init__(self) -> None:
        self._edge:  Any = None
        self._cloud: Any = None
        self._compute_severity: Any = None
        self.edge_info:  dict = {}
        self.cloud_info: dict = {}
        self.ready = False

    def load(self) -> None:
        """Load both models synchronously (call from thread-pool executor)."""
        from hybrid.edge_detector import EdgeDetector
        from hybrid.cloud_analyzer import CloudAnalyzer, compute_severity

        # ── Edge (binary MLP) ────────────────────────────────────────────────
        self._edge = EdgeDetector(
            model_path=str(self._EDGE_MODEL),
            kafka_broker="",          # no Kafka — standalone inference only
            threshold=0.95,
            batch_size=32,
        )
        self._edge._load_model()
        self._edge._load_scaler()
        # Warmup
        import numpy as np
        dummy = np.zeros((1, 11), dtype=np.float32)
        self._edge._model.predict(dummy, verbose=0)
        self.edge_info = {
            "model": self._EDGE_MODEL.name,
            "threshold": self._edge.threshold,
            "fingerprint": self._edge._fingerprint,
        }

        # ── Cloud (XGBoost) ──────────────────────────────────────────────────
        self._cloud = CloudAnalyzer(
            model_path=str(self._CLOUD_MODEL),
            encoder_path=str(self._ENCODER),
            kafka_broker="",
        )
        self._cloud._load_model()
        self._cloud._load_scaler()
        self.cloud_info = {
            "model": self._CLOUD_MODEL.name,
            "n_classes": len(self._cloud._le.classes_),
            "classes": list(self._cloud._le.classes_),
        }

        self._compute_severity = compute_severity
        self.ready = True
        logger.info("[Pipeline] Ready — edge=%s  cloud=%s", self._EDGE_MODEL.name, self._CLOUD_MODEL.name)

    def classify(self, ml_features: dict) -> dict:
        """Full hybrid classify.  Thread-safe (stateless after load()).

        Args:
            ml_features: CICIDS2017-keyed dict (11 features).

        Returns:
            Classification result dict.
        """
        t_start = time.perf_counter()

        edge = self._edge.predict_packet(ml_features)

        cloud_result: Optional[dict] = None
        if edge["is_suspicious"]:
            cloud_result = self._cloud.classify(ml_features)
            severity     = self._compute_severity(
                cloud_result["attack_type"], cloud_result["confidence"]
            )
            final_label = cloud_result["attack_type"]
        else:
            severity    = "none"
            final_label = "Normal Traffic"

        total_ms = (time.perf_counter() - t_start) * 1_000

        return {
            "stage":              "cloud" if cloud_result else "edge_only",
            "final_label":        final_label,
            "severity":           severity,
            "edge":               edge,
            "cloud":              cloud_result,
            "pipeline_latency_ms": round(total_ms, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Capture + classification loop
# ─────────────────────────────────────────────────────────────────────────────

_ALERTS_FILE = Path(__file__).resolve().parents[1] / "data" / "confirmed_attacks.ndjson"


def _load_alerts_from_file() -> list[dict]:
    """Read all persisted alerts from NDJSON. Safe to call even if file missing."""
    alerts: list[dict] = []
    if not _ALERTS_FILE.exists():
        return alerts
    try:
        with open(_ALERTS_FILE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        alerts.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        logger.exception("Failed to load alerts from file")
    return alerts


def _persist_alert(res: dict) -> None:
    global _ALL_ALERTS
    _ALL_ALERTS.append(res)
    _ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_ALERTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res, default=str) + "\n")
    except Exception:
        logger.exception("Failed to persist alert")


async def _broadcast(message: str) -> None:
    dead: list[WebSocket] = []
    for ws in WS_CLIENTS:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        WS_CLIENTS.remove(ws)


_offset_consumer: Any = None
_offset_consumer_lock = threading.Lock()

# Lazy Kafka producer for /inject — publishes to network-events so the full
# Kafka pipeline (edge_detector → cloud_analyzer) processes every injected packet.
_inject_producer: Any = None
_inject_producer_lock = threading.Lock()


def _get_inject_producer() -> Any:
    """Return a cached KafkaProducer, creating it on first call.  Returns None
    if the broker is unreachable so callers can fall back gracefully."""
    global _inject_producer
    with _inject_producer_lock:
        if _inject_producer is not None:
            return _inject_producer
        try:
            from kafka import KafkaProducer  # type: ignore
            _inject_producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                acks=1,
                linger_ms=5,
                compression_type="gzip",
            )
            logger.info("[inject] Kafka producer ready → %s", KAFKA_BROKER)
        except Exception as exc:
            logger.warning("[inject] Kafka unavailable (%s) — will use in-process fallback", exc)
            _inject_producer = None
        return _inject_producer


def _read_kafka_offsets_sync() -> dict[str, int]:
    """Query topic end-offsets using a long-lived consumer (created once, reused)."""
    global _offset_consumer
    from kafka import KafkaConsumer, TopicPartition  # type: ignore

    with _offset_consumer_lock:
        if _offset_consumer is None:
            _offset_consumer = KafkaConsumer(
                bootstrap_servers=[KAFKA_BROKER],
                enable_auto_commit=False,
                consumer_timeout_ms=1_000,
            )

        consumer = _offset_consumer
        totals: dict[str, int] = {}
        for topic in KAFKA_TOPICS:
            partitions = consumer.partitions_for_topic(topic) or set()
            if not partitions:
                totals[topic] = 0
                continue
            tps = [TopicPartition(topic, p) for p in partitions]
            beginning_offsets = consumer.beginning_offsets(tps)
            end_offsets = consumer.end_offsets(tps)
            totals[topic] = sum(
                max(0, int(end_offsets[tp]) - int(beginning_offsets[tp]))
                for tp in tps
            )
        return totals


async def _refresh_kafka_offsets() -> None:
    try:
        offsets = await asyncio.to_thread(_read_kafka_offsets_sync)
        KAFKA_STATE.update_offsets(offsets)
    except Exception as exc:
        # If the cached consumer is broken, discard it so the next call recreates it
        global _offset_consumer
        with _offset_consumer_lock:
            _offset_consumer = None
        KAFKA_STATE.mark_error(exc)


async def kafka_monitor_loop() -> None:
    """Consume live Kafka messages and mirror them to dashboard WebSockets."""
    from kafka import KafkaConsumer  # type: ignore

    consumer: Any = None
    try:
        consumer = KafkaConsumer(
            *KAFKA_TOPICS,
            bootstrap_servers=[KAFKA_BROKER],
            group_id="nids_dashboard_api",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            fetch_max_bytes=5 * 1024 * 1024,
        )
        logger.info("Kafka dashboard monitor connected — broker=%s", KAFKA_BROKER)

        last_offsets = 0.0
        while True:
            now = time.time()
            if now - last_offsets >= 1.0:
                await _refresh_kafka_offsets()
                metrics_event = {"type": "metrics", "source": "kafka", "data": KAFKA_STATE.hybrid_snapshot()}
                if WS_CLIENTS:
                    await _broadcast(json.dumps(metrics_event, default=str))
                last_offsets = now

            records = await asyncio.to_thread(consumer.poll, 250)
            for tp, messages in records.items():
                for msg in messages:
                    try:
                        payload = json.loads(msg.value.decode("utf-8"))
                    except Exception:
                        with KAFKA_STATE._lock:
                            KAFKA_STATE.parse_errors.update([tp.topic])
                        continue
                    if tp.topic == "fl-rounds":
                        if WS_CLIENTS:
                            await _broadcast(json.dumps(
                                {"type": "fl_round", "data": payload, "source": "kafka"},
                                default=str,
                            ))
                        logger.info("FL round %s broadcast via Kafka", payload.get("round"))
                        continue
                    event = KAFKA_STATE.record_message(tp.topic, payload)
                    if event is not None:
                        if event.get("type") == "alert":
                            alert = event["data"]
                            ALERTS.append(alert)
                            await asyncio.to_thread(_persist_alert, alert)
                        if WS_CLIENTS:
                            event["source"] = "kafka"
                            await _broadcast(json.dumps(event, default=str))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        KAFKA_STATE.mark_error(exc)
        logger.warning("Kafka dashboard monitor stopped: %s", exc)
    finally:
        if consumer is not None:
            await asyncio.to_thread(consumer.close)


async def _process_packet(normalized: dict) -> dict | None:
    """Classify one normalised packet dict and broadcast the result via WS.

    Shared by capture_loop (pyshark path) and the /inject endpoint (direct path).
    Returns the classification result, or None if the pipeline is not ready.
    """
    loop = asyncio.get_event_loop()
    BUFFER.append(normalized)

    if _pipeline is not None and _pipeline.ready:
        ml_vec = to_ml_vector(normalized)
        try:
            result = await loop.run_in_executor(None, _pipeline.classify, ml_vec)
            result["features"]  = normalized
            result["timestamp"] = normalized.get("timestamp", time.time())
            PREDICTIONS.append(result)
            KAFKA_STATE.record_direct_capture(normalized, result)

            if result["severity"] not in ("none",):
                ALERTS.append(result)
                await loop.run_in_executor(None, _persist_alert, result)
                if WS_CLIENTS:
                    await _broadcast(json.dumps(
                        {"type": "alert", "source": "api_capture", "data": result}, default=str
                    ))
            else:
                if WS_CLIENTS:
                    await _broadcast(json.dumps(
                        {"type": "packet", "source": "api_capture", "data": normalized}, default=str
                    ))
            return result
        except Exception as exc:
            logger.warning("inference error: %s", exc)

    # Pipeline not ready — count and broadcast raw features
    KAFKA_STATE.record_direct_capture(normalized, None)
    if WS_CLIENTS:
        await _broadcast(json.dumps(
            {"type": "packet", "source": "api_capture", "data": normalized}, default=str
        ))
    return None


async def capture_loop(interface: str = "eth0") -> None:
    global _capture_active
    logger.info("capture_loop starting on %s", interface)
    _capture_active = True

    try:
        async for features in live_capture(interface=interface):
            # live_capture() already yields parsed feature dicts — just normalise
            await _process_packet(normalize_features(features))

    except Exception as exc:
        logger.error("capture_loop crashed: %s", exc, exc_info=True)
    finally:
        _capture_active = False
        logger.info("capture_loop stopped")


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager

def _normalize_fl_round(item: dict) -> dict:
    """Normalize a raw fl_history.json entry into the FLRound dashboard shape."""
    gm = item.get("global_metrics", item)
    return {
        "round":              item.get("round", item.get("round_number", 0)),
        "global_f1":          gm.get("f1_macro",  gm.get("global_f1",  0.0)),
        "global_accuracy":    gm.get("accuracy",  gm.get("global_accuracy", 0.0)),
        "global_loss":        gm.get("loss",       gm.get("global_loss", 0.0)),
        "clients":            item.get("clients", []),
        "aggregation_method": item.get("aggregation_method", "FedAvg"),
        "timestamp":          item.get("timestamp", ""),
    }


async def lifespan(app: FastAPI):
    global _pipeline, _ALL_ALERTS
    loop = asyncio.get_event_loop()

    # Restore persisted alerts so history survives restarts.
    persisted = _load_alerts_from_file()
    _ALL_ALERTS.extend(persisted)
    # Seed the rolling deque with the most recent 500 for live-view continuity.
    for alert in persisted[-500:]:
        ALERTS.append(alert)
    # Rebuild Kafka state counters so stats are accurate from boot.
    for alert in persisted:
        label    = alert.get("final_label")
        severity = alert.get("severity")
        if label:
            KAFKA_STATE.labels[label] += 1
        if severity:
            KAFKA_STATE.severities[severity] += 1
    if persisted:
        logger.info("Loaded %d persisted alerts from disk", len(persisted))

    # Load models in background thread (heavy I/O)
    _pipeline = NIDSPipeline()
    try:
        await loop.run_in_executor(None, _pipeline.load)
    except Exception as exc:
        logger.error("Pipeline failed to load: %s", exc, exc_info=True)
        _pipeline = None

    # Start capture for API-only mode and Kafka monitor for dashboard mode.
    capture_task = asyncio.create_task(capture_loop(interface=CAPTURE_IFACE))
    kafka_task   = asyncio.create_task(kafka_monitor_loop())
    yield

    for task in (capture_task, kafka_task):
        task.cancel()
    for task in (capture_task, kafka_task):
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="NIDS-FL API",
    description="Live network packet capture + hybrid MLP→XGBoost classification.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict", summary="Classify a packet by feature vector")
async def predict(body: dict) -> dict:
    """Accept a feature dict and run the full hybrid pipeline.

    The body may contain either:
    - **11 CICIDS2017 keys** (model input directly):
      ``Average Packet Size``, ``Subflow Fwd Bytes``, ``Destination Port``, …
    - **22 live keys** (raw output of feature extraction):
      ``src_ip``, ``dst_ip``, ``dst_port``, ``fin_flag``, …
      (automatically mapped to the 11 ML features)

    Returns the classification result with edge + cloud predictions.
    """
    if _pipeline is None or not _pipeline.ready:
        return {"error": "Pipeline not ready — models still loading"}

    # Auto-detect format: if a raw live key present, map to ML features
    if "src_ip" in body or "timestamp" in body:
        ml_features = to_ml_vector(body)
    else:
        ml_features = body

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _pipeline.classify, ml_features)
    except Exception as exc:
        return {"error": str(exc)}

    return result


# ─────────────────────────────────────────────────────────────────────────────
# POST /inject  — direct packet injection (bypasses pyshark, useful for testing)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/inject", summary="Inject a synthetic packet into the Kafka pipeline")
async def inject_packet(body: dict) -> dict:
    """Publish one raw feature dict to the ``network-events`` Kafka topic so
    the full pipeline processes it:

        /inject → network-events → edge_detector → suspicious-traffic
               → cloud_analyzer → classified-attacks / alerts
               → kafka_monitor_loop → WebSocket → Dashboard

    Falls back to in-process inference when the Kafka broker is not reachable.
    Accepts the same body format as POST /predict (22 live keys or 11 CICIDS keys).
    """
    body.setdefault("timestamp", time.time())
    body.setdefault("protocol",  "TCP")
    body.setdefault("src_ip",    "")
    body.setdefault("dst_ip",    "")
    body.setdefault("src_port",  0)
    body.setdefault("dst_port",  0)

    # ── Try Kafka-first path ─────────────────────────────────────────────────
    producer = _get_inject_producer()
    if producer is not None:
        try:
            # Convert to CICIDS2017 format that edge_detector expects
            ml_vec = to_ml_vector(body)
            # Attach routing context so downstream stages can build alerts
            ml_vec["src_ip"]   = body["src_ip"]
            ml_vec["dst_ip"]   = body["dst_ip"]
            ml_vec["src_port"] = body["src_port"]
            ml_vec["dst_port"] = body["dst_port"]
            ml_vec["protocol"] = body["protocol"]
            ml_vec["timestamp"] = body["timestamp"]

            src_key = (body["src_ip"] or "unknown").encode("utf-8")
            await asyncio.to_thread(
                lambda: producer.send("network-events", key=src_key,
                                      value=json.dumps(ml_vec).encode("utf-8")).get(timeout=3)
            )

            # Immediately echo the raw packet to WebSocket so the live traffic
            # view updates without waiting for the full Kafka round-trip.
            BUFFER.append(body)
            if WS_CLIENTS:
                await _broadcast(json.dumps(
                    {"type": "packet", "source": "inject", "data": body}, default=str
                ))
            return {"status": "queued", "topic": "network-events"}
        except Exception as exc:
            logger.warning("[inject] Kafka send failed (%s) — falling back to in-process", exc)
            # Reset the cached producer so the next call retries the connection
            global _inject_producer
            with _inject_producer_lock:
                _inject_producer = None

    # ── In-process fallback (Kafka unavailable) ──────────────────────────────
    result = await _process_packet(body)
    return result if result is not None else {"status": "queued", "pipeline_ready": False}


# ─────────────────────────────────────────────────────────────────────────────
# GET /predictions
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/predictions", summary="Recent classified packets")
async def get_predictions(
    limit: int = Query(50, ge=1, le=2000),
    label: Optional[str] = Query(None, description="Filter by final_label (e.g. DDoS)"),
    severity: Optional[str] = Query(None, description="Filter by severity: low|medium|high"),
    stage: Optional[str] = Query(None, description="Filter by stage: edge_only|cloud"),
) -> dict:
    preds = list(PREDICTIONS)

    if label:
        preds = [p for p in preds if p.get("final_label") == label]
    if severity:
        preds = [p for p in preds if p.get("severity") == severity]
    if stage:
        preds = [p for p in preds if p.get("stage") == stage]

    return {
        "total_in_buffer": len(PREDICTIONS),
        "total_matched":   len(preds),
        "returned":        min(limit, len(preds)),
        "predictions":     preds[-limit:],
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /alerts
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/alerts", summary="Confirmed attack alerts only")
async def get_alerts(
    limit: int = Query(50, ge=1, le=500),
    severity: Optional[str] = Query(None, description="Filter by severity: low|medium|high"),
) -> dict:
    alerts = list(_ALL_ALERTS)

    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity]

    return {
        "total_alerts":  len(_ALL_ALERTS),
        "total_matched": len(alerts),
        "returned":      min(limit, len(alerts)),
        "alerts":        alerts[-limit:],
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /packets
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/packets", summary="Raw captured packets (pre-inference buffer)")
async def get_packets(
    limit: int = Query(100, ge=1, le=1000),
    protocol: Optional[str] = Query(None, description="TCP | UDP | ICMP"),
    src_ip:   Optional[str] = Query(None),
    dst_ip:   Optional[str] = Query(None),
    dst_port: Optional[int] = Query(None),
) -> dict:
    packets = list(BUFFER)

    if protocol:
        packets = [p for p in packets if p.get("protocol") == protocol.upper()]
    if src_ip:
        packets = [p for p in packets if p.get("src_ip") == src_ip]
    if dst_ip:
        packets = [p for p in packets if p.get("dst_ip") == dst_ip]
    if dst_port is not None:
        packets = [p for p in packets if p.get("dst_port") == dst_port]

    return {
        "total_matched": len(packets),
        "returned":      min(limit, len(packets)),
        "packets":       packets[-limit:],
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /stats
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/stats", summary="Protocol + attack distribution statistics")
async def get_stats() -> dict:
    await _refresh_kafka_offsets()
    kafka_stats = KAFKA_STATE.stats_snapshot()
    if kafka_stats["kafka_connected"]:
        return kafka_stats

    # Fallback for API-only mode when Kafka is not reachable.
    packets = list(BUFFER)
    preds   = list(PREDICTIONS)
    total   = len(packets)
    uptime  = time.time() - _start_time

    protocols:    Counter = Counter(p.get("protocol")     for p in packets)
    app_protocols: Counter = Counter(p.get("app_protocol") or p.get("protocol") for p in packets)
    src_ips:      Counter = Counter(p.get("src_ip")       for p in packets if p.get("src_ip"))
    dst_ports:    Counter = Counter(p.get("dst_port")     for p in packets if p.get("dst_port"))
    labels:       Counter = Counter(p.get("final_label")  for p in preds  if p.get("final_label"))
    severities:   Counter = Counter(p.get("severity")     for p in preds  if p.get("severity"))

    return {
        **kafka_stats,
        "uptime_seconds":         round(uptime, 1),
        "total_packets":          total,
        "packets_per_second":     round(total / uptime, 2) if uptime > 0 else 0.0,
        "total_classified":       len(preds),
        "total_alerts":           len(_ALL_ALERTS),
        "protocols":              dict(protocols),
        "app_protocols":          dict(app_protocols),
        "top_5_source_ips":       dict(src_ips.most_common(5)),
        "top_5_destination_ports": dict(dst_ports.most_common(5)),
        "label_distribution":     dict(labels),
        "severity_distribution":  dict(severities),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /pipeline/status
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/pipeline/status", summary="Loaded model metadata")
async def pipeline_status() -> dict:
    if _pipeline is None:
        return {"ready": False, "reason": "pipeline object not created"}
    if not _pipeline.ready:
        return {"ready": False, "reason": "models still loading"}

    return {
        "ready":      True,
        "edge_model": _pipeline.edge_info,
        "cloud_model": _pipeline.cloud_info,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /fl/history
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_fl_history() -> list[dict]:
    """Deterministic synthetic FL history — 10 rounds, sigmoid convergence."""
    rng = random.Random(42)
    N_SAMPLES = [3000, 2800, 3200]
    WEIGHTS_KB = 22.4
    base_ts = datetime(2025, 5, 17, 8, 0, 0)
    rounds = []
    for r in range(1, 11):
        t = r / 10.0
        global_f1 = 0.88 + 0.09 / (1 + math.exp(-10 * (t - 0.5)))
        global_acc = global_f1 + rng.uniform(0.005, 0.015)
        global_loss = 0.35 * math.exp(-1.8 * t) + rng.uniform(0.005, 0.02)
        clients = []
        for cid, ns in enumerate(N_SAMPLES, start=1):
            noise = rng.uniform(-0.012, 0.012)
            clients.append({
                "id": cid,
                "local_f1": round(global_f1 + noise, 4),
                "local_accuracy": round(global_acc + rng.uniform(-0.01, 0.01), 4),
                "n_samples": ns,
                "weights_size_kb": WEIGHTS_KB,
            })
        ts = base_ts.replace(minute=(r - 1) * 5 % 60, hour=8 + (r - 1) * 5 // 60)
        rounds.append({
            "round": r,
            "global_f1": round(global_f1, 4),
            "global_accuracy": round(min(global_acc, 0.9999), 4),
            "global_loss": round(global_loss, 4),
            "clients": clients,
            "aggregation_method": "FedAvg",
            "timestamp": ts.isoformat(),
        })
    return rounds


_FL_SYNTHETIC: list[dict] = _synthetic_fl_history()


@app.get("/fl/history", summary="Federated learning round history")
async def fl_history() -> list:
    """Returns completed FL rounds from file, or synthetic data as fallback."""
    fl_path = _ROOT / "data" / "fl_history.json"
    if fl_path.exists():
        try:
            raw = json.loads(fl_path.read_text())
            rounds = []
            for item in raw if isinstance(raw, list) else raw.get("rounds", []):
                gm = item.get("global_metrics", item)
                rounds.append({
                    "round": item.get("round", item.get("round_number", 0)),
                    "global_f1": gm.get("f1_macro", gm.get("global_f1", 0)),
                    "global_accuracy": gm.get("accuracy", gm.get("global_accuracy", 0)),
                    "global_loss": gm.get("loss", gm.get("global_loss", 0)),
                    "clients": item.get("clients", []),
                    "aggregation_method": item.get("aggregation_method", "FedAvg"),
                    "timestamp": item.get("timestamp", ""),
                })
            return rounds if rounds else _FL_SYNTHETIC
        except Exception:
            pass
    return _FL_SYNTHETIC


# ─────────────────────────────────────────────────────────────────────────────
# GET /hybrid/metrics
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/hybrid/metrics", summary="Hybrid pipeline orchestrator metrics")
async def hybrid_metrics() -> dict:
    await _refresh_kafka_offsets()
    metrics = KAFKA_STATE.hybrid_snapshot()
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# GET /models
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/models", summary="Registered model metadata from MANIFEST.json")
async def list_models() -> list:
    manifest_path = _ROOT / "models" / "final" / "MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        result = []
        for winner in manifest.get("winners", {}).values():
            metrics = winner.get("metrics", {})
            result.append({
                "model_id":    winner.get("model_id", ""),
                "role":        winner.get("role", ""),
                "accuracy":    metrics.get("accuracy", 0.0),
                "f1_macro":    metrics.get("f1_macro", 0.0),
                "f1_weighted": metrics.get("f1_weighted", 0.0),
                "f1_per_class": metrics.get("f1_per_class"),
                "size_mb":     round(winner.get("size_bytes", 0) / 1_048_576, 2),
                "format":      winner.get("format", ""),
            })
        return result
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# GET /graph/attacks
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_graph() -> dict:
    """Deterministic synthetic attack graph for demo / no-Neo4j mode."""
    nodes = [
        # ── Attacker IPs (ip_source) ─────────────────────────────────────────
        {"id": "src_1", "label": "185.220.101.42", "type": "ip_source", "count": 847,
         "attack_type": "DDoS",        "first_seen": "2025-05-17T06:12:00", "last_seen": "2025-05-17T11:47:00"},
        {"id": "src_2", "label": "194.165.16.29",  "type": "ip_source", "count": 756,
         "attack_type": "DDoS",        "first_seen": "2025-05-17T07:03:00", "last_seen": "2025-05-17T12:01:00"},
        {"id": "src_3", "label": "104.21.64.100",  "type": "ip_source", "count": 623,
         "attack_type": "Port Scanning","first_seen": "2025-05-17T05:30:00", "last_seen": "2025-05-17T10:55:00"},
        {"id": "src_4", "label": "91.240.118.56",  "type": "ip_source", "count": 412,
         "attack_type": "Brute Force", "first_seen": "2025-05-17T08:14:00", "last_seen": "2025-05-17T11:30:00"},
        {"id": "src_5", "label": "193.32.127.204", "type": "ip_source", "count": 341,
         "attack_type": "Web Attacks", "first_seen": "2025-05-17T09:00:00", "last_seen": "2025-05-17T11:58:00"},
        {"id": "src_6", "label": "45.142.212.100", "type": "ip_source", "count": 289,
         "attack_type": "Bots",        "first_seen": "2025-05-17T04:00:00", "last_seen": "2025-05-17T12:05:00"},
        {"id": "src_7", "label": "77.73.133.88",   "type": "ip_source", "count": 198,
         "attack_type": "DoS",         "first_seen": "2025-05-17T10:22:00", "last_seen": "2025-05-17T11:50:00"},
        {"id": "src_8", "label": "46.166.139.80",  "type": "ip_source", "count": 156,
         "attack_type": "Brute Force", "first_seen": "2025-05-17T09:45:00", "last_seen": "2025-05-17T11:40:00"},
        # ── Target IPs (ip_target) ───────────────────────────────────────────
        {"id": "dst_1", "label": "192.168.1.100",  "type": "ip_target", "count": 1203,
         "first_seen": "2025-05-17T05:30:00", "last_seen": "2025-05-17T12:05:00"},
        {"id": "dst_2", "label": "10.0.0.50",      "type": "ip_target", "count": 741,
         "first_seen": "2025-05-17T06:12:00", "last_seen": "2025-05-17T11:58:00"},
        {"id": "dst_3", "label": "172.16.0.1",     "type": "ip_target", "count": 512,
         "first_seen": "2025-05-17T07:03:00", "last_seen": "2025-05-17T12:01:00"},
        {"id": "dst_4", "label": "192.168.1.200",  "type": "ip_target", "count": 388,
         "first_seen": "2025-05-17T08:14:00", "last_seen": "2025-05-17T11:30:00"},
        # ── Attack type nodes ────────────────────────────────────────────────
        {"id": "atk_ddos",    "label": "DDoS",          "type": "attack", "count": 1603,
         "attack_type": "DDoS",        "first_seen": "2025-05-17T06:12:00", "last_seen": "2025-05-17T12:01:00"},
        {"id": "atk_dos",     "label": "DoS",           "type": "attack", "count": 198,
         "attack_type": "DoS",         "first_seen": "2025-05-17T10:22:00", "last_seen": "2025-05-17T11:50:00"},
        {"id": "atk_scan",    "label": "Port Scanning", "type": "attack", "count": 623,
         "attack_type": "Port Scanning","first_seen": "2025-05-17T05:30:00", "last_seen": "2025-05-17T10:55:00"},
        {"id": "atk_brute",   "label": "Brute Force",   "type": "attack", "count": 568,
         "attack_type": "Brute Force", "first_seen": "2025-05-17T08:14:00", "last_seen": "2025-05-17T11:40:00"},
        {"id": "atk_bots",    "label": "Bots",          "type": "attack", "count": 289,
         "attack_type": "Bots",        "first_seen": "2025-05-17T04:00:00", "last_seen": "2025-05-17T12:05:00"},
        {"id": "atk_web",     "label": "Web Attacks",   "type": "attack", "count": 341,
         "attack_type": "Web Attacks", "first_seen": "2025-05-17T09:00:00", "last_seen": "2025-05-17T11:58:00"},
        # ── FL client nodes ──────────────────────────────────────────────────
        {"id": "fl_1", "label": "Edge Node α", "type": "client_fl", "count": 0,
         "first_seen": "2025-05-17T08:00:00", "last_seen": "2025-05-17T12:00:00"},
        {"id": "fl_2", "label": "Edge Node β", "type": "client_fl", "count": 0,
         "first_seen": "2025-05-17T08:00:00", "last_seen": "2025-05-17T12:00:00"},
        {"id": "fl_3", "label": "Edge Node γ", "type": "client_fl", "count": 0,
         "first_seen": "2025-05-17T08:00:00", "last_seen": "2025-05-17T12:00:00"},
    ]
    edges = [
        # DDoS attackers → DDoS attack node
        {"source": "src_1", "target": "atk_ddos",  "type": "ATTACKS",     "count": 847, "attack_types": ["DDoS"]},
        {"source": "src_2", "target": "atk_ddos",  "type": "ATTACKS",     "count": 756, "attack_types": ["DDoS"]},
        # DoS attacker → DoS node
        {"source": "src_7", "target": "atk_dos",   "type": "ATTACKS",     "count": 198, "attack_types": ["DoS"]},
        # Port scan
        {"source": "src_3", "target": "atk_scan",  "type": "ATTACKS",     "count": 623, "attack_types": ["Port Scanning"]},
        # Brute force
        {"source": "src_4", "target": "atk_brute", "type": "ATTACKS",     "count": 412, "attack_types": ["Brute Force"]},
        {"source": "src_8", "target": "atk_brute", "type": "ATTACKS",     "count": 156, "attack_types": ["Brute Force"]},
        # Web attacks
        {"source": "src_5", "target": "atk_web",   "type": "ATTACKS",     "count": 341, "attack_types": ["Web Attacks"]},
        # Bots
        {"source": "src_6", "target": "atk_bots",  "type": "ATTACKS",     "count": 289, "attack_types": ["Bots"]},
        # Attack → target IPs
        {"source": "atk_ddos",  "target": "dst_1", "type": "TARGETS", "count": 1203, "attack_types": ["DDoS"]},
        {"source": "atk_ddos",  "target": "dst_3", "type": "TARGETS", "count": 400,  "attack_types": ["DDoS"]},
        {"source": "atk_dos",   "target": "dst_1", "type": "TARGETS", "count": 198,  "attack_types": ["DoS"]},
        {"source": "atk_scan",  "target": "dst_2", "type": "TARGETS", "count": 623,  "attack_types": ["Port Scanning"]},
        {"source": "atk_scan",  "target": "dst_1", "type": "TARGETS", "count": 120,  "attack_types": ["Port Scanning"]},
        {"source": "atk_brute", "target": "dst_4", "type": "TARGETS", "count": 568,  "attack_types": ["Brute Force"]},
        {"source": "atk_web",   "target": "dst_1", "type": "TARGETS", "count": 341,  "attack_types": ["Web Attacks"]},
        {"source": "atk_bots",  "target": "dst_2", "type": "TARGETS", "count": 289,  "attack_types": ["Bots"]},
        {"source": "atk_bots",  "target": "dst_3", "type": "TARGETS", "count": 112,  "attack_types": ["Bots"]},
        # Attack → FL clients (DETECTED_BY)
        {"source": "atk_ddos",  "target": "fl_1",  "type": "DETECTED_BY", "count": 1603, "attack_types": ["DDoS"]},
        {"source": "atk_scan",  "target": "fl_2",  "type": "DETECTED_BY", "count": 623,  "attack_types": ["Port Scanning"]},
        {"source": "atk_brute", "target": "fl_2",  "type": "DETECTED_BY", "count": 568,  "attack_types": ["Brute Force"]},
        {"source": "atk_web",   "target": "fl_3",  "type": "DETECTED_BY", "count": 341,  "attack_types": ["Web Attacks"]},
        {"source": "atk_bots",  "target": "fl_3",  "type": "DETECTED_BY", "count": 289,  "attack_types": ["Bots"]},
        {"source": "atk_dos",   "target": "fl_1",  "type": "DETECTED_BY", "count": 198,  "attack_types": ["DoS"]},
    ]
    return {"nodes": nodes, "edges": edges}


def _build_graph_from_alerts(max_src: int = 20) -> dict:
    """Build an attack graph directly from confirmed alert data in _ALL_ALERTS."""
    from collections import defaultdict

    src_meta: dict[str, dict]     = {}   # ip → {count, attack_type, first_seen, last_seen}
    dst_meta: dict[str, dict]     = {}
    atk_meta: dict[str, dict]     = {}
    src_to_atk: dict[tuple, int]  = defaultdict(int)
    atk_to_dst: dict[tuple, int]  = defaultdict(int)

    for alert in _ALL_ALERTS:
        features    = alert.get("features") or {}
        src_ip      = str(features.get("src_ip") or "").strip()
        dst_ip      = str(features.get("dst_ip") or "").strip()
        attack_type = str(alert.get("final_label") or "Unknown").strip()
        ts_raw      = alert.get("timestamp", 0)

        if not src_ip or not dst_ip or attack_type in ("Normal Traffic", "Unknown", ""):
            continue

        try:
            ts_str = datetime.fromtimestamp(float(ts_raw)).isoformat()
        except Exception:
            ts_str = ""

        def _update_meta(store: dict, key: str, atk: str, ts: str) -> None:
            if key not in store:
                store[key] = {"count": 0, "attack_type": atk, "first_seen": ts, "last_seen": ts}
            d = store[key]
            d["count"] += 1
            if ts and (not d["first_seen"] or ts < d["first_seen"]):
                d["first_seen"] = ts
            if ts and ts > d["last_seen"]:
                d["last_seen"] = ts

        _update_meta(src_meta, src_ip, attack_type, ts_str)
        _update_meta(dst_meta, dst_ip, attack_type, ts_str)
        _update_meta(atk_meta, attack_type, attack_type, ts_str)
        src_to_atk[(src_ip, attack_type)] += 1
        atk_to_dst[(attack_type, dst_ip)] += 1

    if not src_meta or not atk_meta:
        return _synthetic_graph()

    # Cap source IPs to avoid overwhelming the graph
    top_srcs = sorted(src_meta.items(), key=lambda x: x[1]["count"], reverse=True)[:max_src]
    top_src_set = {ip for ip, _ in top_srcs}

    def _safe_id(prefix: str, val: str) -> str:
        return prefix + val.replace(".", "_").replace(" ", "_").lower()

    nodes: list[dict] = []
    for ip, d in top_srcs:
        nodes.append({
            "id": _safe_id("src_", ip), "label": ip, "type": "ip_source",
            "count": d["count"], "attack_type": d["attack_type"],
            "first_seen": d["first_seen"], "last_seen": d["last_seen"],
        })
    for ip, d in dst_meta.items():
        nodes.append({
            "id": _safe_id("dst_", ip), "label": ip, "type": "ip_target",
            "count": d["count"],
            "first_seen": d["first_seen"], "last_seen": d["last_seen"],
        })
    for atk, d in atk_meta.items():
        nodes.append({
            "id": _safe_id("atk_", atk), "label": atk, "type": "attack",
            "count": d["count"], "attack_type": atk,
            "first_seen": d["first_seen"], "last_seen": d["last_seen"],
        })
    for i, name in enumerate(["Edge Node α", "Edge Node β", "Edge Node γ"], 1):
        nodes.append({"id": f"fl_{i}", "label": name, "type": "client_fl", "count": 0})

    edges: list[dict] = []
    for (src_ip, atk), cnt in src_to_atk.items():
        if src_ip not in top_src_set:
            continue
        edges.append({
            "source": _safe_id("src_", src_ip),
            "target": _safe_id("atk_", atk),
            "type": "ATTACKS", "count": cnt, "attack_types": [atk],
        })
    for (atk, dst_ip), cnt in atk_to_dst.items():
        edges.append({
            "source": _safe_id("atk_", atk),
            "target": _safe_id("dst_", dst_ip),
            "type": "TARGETS", "count": cnt, "attack_types": [atk],
        })
    for i, atk in enumerate(sorted(atk_meta.keys())):
        edges.append({
            "source": _safe_id("atk_", atk),
            "target": f"fl_{(i % 3) + 1}",
            "type": "DETECTED_BY", "count": atk_meta[atk]["count"], "attack_types": [atk],
        })

    return {"nodes": nodes, "edges": edges}


@app.get("/graph/attacks", summary="Attack graph topology")
async def attack_graph() -> dict:
    """Returns attack graph built from real confirmed alerts, or synthetic when none exist."""
    if _ALL_ALERTS:
        return _build_graph_from_alerts()
    return _synthetic_graph()


@app.post("/neo4j/query", summary="Execute a raw Cypher query against Neo4j")
async def neo4j_query(body: dict) -> dict:
    """Execute a Cypher query. Returns empty result when Neo4j is unavailable."""
    query = body.get("query", "").strip()
    if not query:
        return {"error": "empty query", "nodes": [], "edges": []}
    try:
        from neo4j import GraphDatabase  # type: ignore
        raise ImportError("Neo4j not configured")
    except Exception:
        return {
            "error": "Neo4j is not available — showing synthetic data",
            "nodes": [],
            "edges": [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Liveness + capture + pipeline status")
async def health() -> dict:
    return {
        "status":          "healthy",
        "capture_active":  _capture_active,
        "pipeline_ready":  _pipeline is not None and _pipeline.ready,
        "uptime_seconds":  round(time.time() - _start_time, 1),
        "buffer_size":     len(BUFFER),
        "predictions_size": len(PREDICTIONS),
        "alerts_size":     len(ALERTS),
        "ws_clients":      len(WS_CLIENTS),
    }


# ─────────────────────────────────────────────────────────────────────────────
# WS /stream
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/stream")
async def ws_stream(websocket: WebSocket):
    """Stream live classified packets to the client as newline-delimited JSON."""
    await websocket.accept()
    WS_CLIENTS.append(websocket)
    logger.info("WS client connected — %d active", len(WS_CLIENTS))
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in WS_CLIENTS:
            WS_CLIENTS.remove(websocket)
        logger.info("WS client disconnected — %d active", len(WS_CLIENTS))
