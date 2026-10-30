"""
hybrid/orchestrator.py
-----------------------
Pipeline monitor / metrics aggregator for the hybrid NIDS pipeline.

Consumes all 4 Kafka topics simultaneously with a single consumer group
(group_id="nids_orchestrator") using auto_offset_reset="latest" so it
observes live traffic without replaying history:

    network-events      → packets_total
    suspicious-traffic  → packets_forwarded_to_cloud, avg_edge_latency_ms
    classified-attacks  → attacks_confirmed, attack_type_distribution, latencies
    alerts              → alerts_published (consistency check)

Metrics are saved to data/hybrid_metrics.json every save_interval seconds
(default 10 s) using an atomic rename so readers never see a partial file.

An optional Prometheus-style /metrics endpoint is served on
http://localhost:<port>/metrics via stdlib http.server (no Flask needed).
Stop it with --no-prometheus.

ASCII dashboard is printed to stdout at every save interval.

Usage:
    python -m hybrid.orchestrator
    python -m hybrid.orchestrator --broker localhost:29092 --interval 10
    python -m hybrid.orchestrator --metrics-port 9200
    python -m hybrid.orchestrator --no-prometheus
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Topics ────────────────────────────────────────────────────────────────────

TOPIC_EVENTS     = "network-events"
TOPIC_SUSPICIOUS = "suspicious-traffic"
TOPIC_CLASSIFIED = "classified-attacks"
TOPIC_ALERTS     = "alerts"

_ALL_TOPICS = [TOPIC_EVENTS, TOPIC_SUSPICIOUS, TOPIC_CLASSIFIED, TOPIC_ALERTS]

# ── Labels treated as non-attack by cloud_analyzer ───────────────────────────

_BENIGN_LABELS = {"Normal Traffic", "BENIGN"}

# ── Canonical output path ─────────────────────────────────────────────────────

_DATA_DIR     = _ROOT / "data"
_METRICS_PATH = _DATA_DIR / "hybrid_metrics.json"

# ── Rolling-window size for latency percentile ────────────────────────────────

_LAT_MAXLEN = 10_000

# ── Logging ───────────────────────────────────────────────────────────────────

_ANSI = {
    "DEBUG":   "\033[36m",
    "INFO":    "\033[32m",   # green for 🟢 orchestrator
    "WARNING": "\033[33m",
    "ERROR":   "\033[31m",
    "RESET":   "\033[0m",
}


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _ANSI.get(record.levelname, _ANSI["RESET"])
        record.levelname = f"{color}{record.levelname:7}{_ANSI['RESET']}"
        return super().format(record)


logger = logging.getLogger(__name__)
_PREFIX = "🟢 [Orchestrator]"


def _log(msg: str, level: str = "info") -> None:
    getattr(logger, level)("%s %s", _PREFIX, msg)


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus HTTP handler (stdlib only)
# ─────────────────────────────────────────────────────────────────────────────

class _PrometheusHandler(BaseHTTPRequestHandler):
    """Single-endpoint HTTP handler — serves GET /metrics in Prometheus format.

    The owning HTTPServer instance must carry an ``orchestrator`` attribute
    (injected in _start_prometheus_server) so the handler can call
    _render_prometheus() without keeping a global reference.
    """

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return

        body = self.server.orchestrator._render_prometheus()  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: Any) -> None:  # suppress default access log to stdout
        pass


# ─────────────────────────────────────────────────────────────────────────────
# HybridOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class HybridOrchestrator:
    """Pipeline monitor — consumes all 4 Kafka topics and aggregates metrics.

    All counter/latency fields are protected by ``_lock`` so that the
    Prometheus thread and the main poll loop never race on reads/writes.

    Args:
        kafka_broker:   Kafka bootstrap address (host:port).
        save_interval:  Seconds between dashboard refresh and JSON saves.
        metrics_port:   Port for the Prometheus HTTP endpoint (None = disabled).
    """

    def __init__(
        self,
        kafka_broker:  str = "localhost:29092",
        save_interval: float = 10.0,
        metrics_port:  Optional[int] = 9100,
    ) -> None:
        self.kafka_broker  = kafka_broker
        self.save_interval = save_interval
        self.metrics_port  = metrics_port

        self._consumer: Any  = None
        self._lock = threading.Lock()

        # ── Cumulative counters ───────────────────────────────────────────────
        self._packets_total     = 0
        self._packets_forwarded = 0
        self._attacks_confirmed = 0
        self._alerts_received   = 0
        self._parse_errors: Dict[str, int] = {t: 0 for t in _ALL_TOPICS}

        # ── Rolling latency windows (maxlen prevents unbounded growth) ─────────
        self._edge_latencies:  Deque[float] = deque(maxlen=_LAT_MAXLEN)
        self._cloud_latencies: Deque[float] = deque(maxlen=_LAT_MAXLEN)
        self._total_latencies: Deque[float] = deque(maxlen=_LAT_MAXLEN)

        # ── Attack type distribution ──────────────────────────────────────────
        self._attack_counts: Dict[str, int] = {}

        # ── Timing ───────────────────────────────────────────────────────────
        self._start_time = time.time()
        self._last_save  = time.time()

    # ── Kafka setup ───────────────────────────────────────────────────────────

    def _init_kafka(self) -> None:
        from kafka import KafkaConsumer  # type: ignore

        self._consumer = KafkaConsumer(
            *_ALL_TOPICS,
            bootstrap_servers=[self.kafka_broker],
            group_id="nids_orchestrator",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            fetch_max_bytes=5 * 1024 * 1024,
        )
        # Lock in 'latest' offset before traffic starts flowing
        self._consumer.poll(timeout_ms=300)
        _log(
            f"Kafka connecté  broker={self.kafka_broker}  "
            f"topics={_ALL_TOPICS}"
        )

    # ── Per-topic message handlers ────────────────────────────────────────────

    def _handle_events(self, msg: Any) -> None:
        """Count raw packets arriving on network-events."""
        with self._lock:
            self._packets_total += 1

    def _handle_suspicious(self, msg: Any) -> None:
        """Count packets forwarded to the cloud stage; collect edge latency."""
        payload: Dict[str, Any] = json.loads(msg.value.decode("utf-8"))
        edge_pred = payload.get("edge_prediction", {})
        lat = float(edge_pred.get("edge_latency_ms", 0.0))

        with self._lock:
            self._packets_forwarded += 1
            if lat > 0.0:
                self._edge_latencies.append(lat)

    def _handle_classified(self, msg: Any) -> None:
        """Count confirmed attacks; collect cloud + total latencies; update distribution."""
        payload: Dict[str, Any] = json.loads(msg.value.decode("utf-8"))
        cloud       = payload.get("cloud_prediction", {})
        attack_type = cloud.get("attack_type", "Unknown")
        cloud_lat   = float(cloud.get("cloud_latency_ms", 0.0))
        total_lat   = float(payload.get("pipeline_total_latency_ms", 0.0))

        with self._lock:
            if attack_type not in _BENIGN_LABELS:
                self._attacks_confirmed += 1
            if cloud_lat > 0.0:
                self._cloud_latencies.append(cloud_lat)
            if total_lat > 0.0:
                self._total_latencies.append(total_lat)
            self._attack_counts[attack_type] = (
                self._attack_counts.get(attack_type, 0) + 1
            )

    def _handle_alerts(self, msg: Any) -> None:
        """Track confirmed alerts published to the alerting bus."""
        with self._lock:
            self._alerts_received += 1

    # Dispatch table — maps Kafka topic name → handler (called as handler(self, msg))
    _HANDLERS: Dict[str, Any] = {
        TOPIC_EVENTS:     _handle_events,
        TOPIC_SUSPICIOUS: _handle_suspicious,
        TOPIC_CLASSIFIED: _handle_classified,
        TOPIC_ALERTS:     _handle_alerts,
    }

    # ── Metrics computation ───────────────────────────────────────────────────

    def collect_metrics(self) -> dict:
        """Return a thread-safe snapshot of all pipeline metrics.

        Counters are copied under the lock; derived stats (means, percentile)
        are computed outside the lock to minimise contention.

        Returns:
            JSON-serialisable dict with all pipeline KPIs.
        """
        with self._lock:
            total     = self._packets_total
            fwd       = self._packets_forwarded
            confirmed = self._attacks_confirmed
            alerts    = self._alerts_received
            edge_lats  = list(self._edge_latencies)
            cloud_lats = list(self._cloud_latencies)
            total_lats = list(self._total_latencies)
            attack_dist  = dict(self._attack_counts)
            parse_errors = dict(self._parse_errors)

        # Derived rates (computed outside lock)
        bw_saved = 1.0 - (fwd / total) if total > 0 else 0.0
        fp_rate  = max(0.0, (fwd - confirmed) / fwd) if fwd > 0 else 0.0

        avg_edge  = float(np.mean(edge_lats))  if edge_lats  else 0.0
        avg_cloud = float(np.mean(cloud_lats)) if cloud_lats else 0.0
        p95_total = float(np.percentile(total_lats, 95)) if total_lats else 0.0

        return {
            "timestamp":                    datetime.now(timezone.utc).isoformat(),
            "uptime_seconds":               round(time.time() - self._start_time, 1),
            "packets_total":                total,
            "packets_forwarded_to_cloud":   fwd,
            "attacks_confirmed_by_xgboost": confirmed,
            "alerts_published":             alerts,
            "edge_false_positive_rate":     round(fp_rate, 4),
            "bandwidth_saved_pct":          round(bw_saved * 100, 2),
            "avg_edge_latency_ms":          round(avg_edge, 2),
            "avg_cloud_latency_ms":         round(avg_cloud, 2),
            "p95_total_latency_ms":         round(p95_total, 2),
            "attack_type_distribution":     attack_dist,
            "parse_errors":                 parse_errors,
        }

    # ── JSON persistence ──────────────────────────────────────────────────────

    def _save_metrics(self, metrics: dict) -> None:
        """Atomically write metrics to data/hybrid_metrics.json (rename trick)."""
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _METRICS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_METRICS_PATH)   # atomic on POSIX

    # ── ASCII dashboard ───────────────────────────────────────────────────────

    def _print_dashboard(self, m: dict) -> None:
        """Print a compact, box-drawn ASCII dashboard to stdout."""
        W = 64   # inner width (between ║ chars); total line = W + 2

        def _bar(count: int, total: int, width: int = 18) -> str:
            filled = int(width * count / total) if total else 0
            return "█" * filled + "░" * (width - filled)

        def _row(content: str) -> str:
            return "║" + content.ljust(W) + "║"

        def _sep(left: str = "╠", fill: str = "═", right: str = "╣") -> str:
            return left + fill * W + right

        # Non-benign attacks for the distribution panel
        attacks = {
            k: v for k, v in m["attack_type_distribution"].items()
            if k not in _BENIGN_LABELS and v > 0
        }
        total_attacks = sum(attacks.values()) or 1

        header = "NIDS Hybrid Pipeline — Live Metrics"
        lines = [
            _sep("╔", "═", "╗"),
            _row(f"  {header}".center(W)),
            _sep(),
            _row(f"  🕐 uptime {m['uptime_seconds']:>8.0f} s"),
            _sep("╠", "─", "╣"),
            _row("  TRAFFIC"),
            _row(f"    total          {m['packets_total']:>14,}"),
            _row(f"    forwarded       {m['packets_forwarded_to_cloud']:>13,}  → cloud"),
            _row(f"    bandwidth saved {m['bandwidth_saved_pct']:>12.1f} %"),
            _sep("╠", "─", "╣"),
            _row("  DETECTION"),
            _row(f"    attacks confirmed  {m['attacks_confirmed_by_xgboost']:>10,}"),
            _row(f"    alerts published   {m['alerts_published']:>10,}"),
            _row(f"    edge FP rate       {m['edge_false_positive_rate'] * 100:>9.1f} %"),
            _sep("╠", "─", "╣"),
            _row("  LATENCY"),
            _row(f"    edge  avg   {m['avg_edge_latency_ms']:>10.2f} ms"),
            _row(f"    cloud avg   {m['avg_cloud_latency_ms']:>10.2f} ms"),
            _row(f"    p95 total   {m['p95_total_latency_ms']:>10.2f} ms"),
            _sep("╠", "─", "╣"),
            _row("  ATTACK DISTRIBUTION"),
        ]

        if attacks:
            top5 = sorted(attacks.items(), key=lambda kv: -kv[1])[:5]
            for name, count in top5:
                pct    = 100.0 * count / total_attacks
                b      = _bar(count, total_attacks)
                content = f"  {name:<16} {b} {count:>6} ({pct:4.1f}%)"
                lines.append(_row(content))
        else:
            lines.append(_row("    (aucune attaque détectée)"))

        lines.append(_sep("╚", "═", "╝"))

        print("\n" + "\n".join(lines), flush=True)

    # ── Prometheus rendering ──────────────────────────────────────────────────

    def _render_prometheus(self) -> bytes:
        """Build a Prometheus exposition-format payload from the current snapshot.

        Called from the HTTP server thread — collect_metrics() is thread-safe.
        """
        m = self.collect_metrics()
        lines: List[str] = []

        def _gauge(name: str, help_text: str, value: float) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        def _counter(name: str, help_text: str, value: float) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        _counter("nids_packets_total",
                 "Total raw packets observed on network-events",
                 m["packets_total"])
        _counter("nids_packets_forwarded_to_cloud",
                 "Packets forwarded by edge MLP to cloud stage",
                 m["packets_forwarded_to_cloud"])
        _counter("nids_attacks_confirmed_by_xgboost",
                 "Packets confirmed as attacks by XGBoost (non-BENIGN)",
                 m["attacks_confirmed_by_xgboost"])
        _counter("nids_alerts_published",
                 "Alerts published to the alerts topic",
                 m["alerts_published"])
        _gauge("nids_edge_false_positive_rate",
               "Edge MLP false positive rate: (forwarded - confirmed) / forwarded",
               m["edge_false_positive_rate"])
        _gauge("nids_bandwidth_saved_pct",
               "Percentage of traffic dropped by the edge filter (%)",
               m["bandwidth_saved_pct"])
        _gauge("nids_avg_edge_latency_ms",
               "Rolling average edge MLP inference latency (ms)",
               m["avg_edge_latency_ms"])
        _gauge("nids_avg_cloud_latency_ms",
               "Rolling average XGBoost classification latency (ms)",
               m["avg_cloud_latency_ms"])
        _gauge("nids_p95_total_latency_ms",
               "95th-percentile end-to-end pipeline latency over last 10 000 packets (ms)",
               m["p95_total_latency_ms"])
        _counter("nids_uptime_seconds",
                 "Orchestrator uptime in seconds",
                 m["uptime_seconds"])

        # Per-attack-type labelled gauge
        lines.append("# HELP nids_attack_count Packets classified per attack type")
        lines.append("# TYPE nids_attack_count gauge")
        for attack_type, count in m["attack_type_distribution"].items():
            # Escape label value (replace " → standard Prometheus label encoding)
            safe = attack_type.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'nids_attack_count{{type="{safe}"}} {count}')

        lines.append("")   # required trailing newline
        return "\n".join(lines).encode("utf-8")

    # ── Prometheus HTTP server ────────────────────────────────────────────────

    def _start_prometheus_server(self) -> None:
        """Launch the Prometheus HTTP server in a daemon thread."""
        if self.metrics_port is None:
            return

        server = HTTPServer(("0.0.0.0", self.metrics_port), _PrometheusHandler)
        server.orchestrator = self  # type: ignore[attr-defined]

        t = threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name="prometheus-http",
        )
        t.start()
        _log(f"Prometheus endpoint  →  http://localhost:{self.metrics_port}/metrics")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the orchestrator.  Blocks until interrupted (Ctrl-C)."""
        _log(
            f"Démarrage  broker={self.kafka_broker}  "
            f"save_interval={self.save_interval}s  "
            f"metrics_port={self.metrics_port or 'disabled'}"
        )

        self._init_kafka()

        if self.metrics_port is not None:
            self._start_prometheus_server()

        _log(f"En écoute sur {_ALL_TOPICS} …")

        try:
            while True:
                records = self._consumer.poll(timeout_ms=200)

                for tp, messages in records.items():
                    handler = self._HANDLERS.get(tp.topic)
                    if handler is None:
                        continue
                    for msg in messages:
                        try:
                            handler(self, msg)
                        except (KeyError, ValueError, json.JSONDecodeError) as exc:
                            with self._lock:
                                self._parse_errors[tp.topic] = (
                                    self._parse_errors.get(tp.topic, 0) + 1
                                )
                            logger.debug(
                                "%s Parse error [%s]: %s", _PREFIX, tp.topic, exc
                            )

                # ── Periodic dashboard + save ─────────────────────────────────
                now = time.time()
                if now - self._last_save >= self.save_interval:
                    metrics = self.collect_metrics()
                    self._save_metrics(metrics)
                    self._print_dashboard(metrics)
                    _log(f"Métriques sauvegardées → {_METRICS_PATH}")
                    self._last_save = now

        except KeyboardInterrupt:
            _log("Arrêt demandé (Ctrl-C)")
        finally:
            if self._consumer:
                self._consumer.close()
            metrics = self.collect_metrics()
            self._save_metrics(metrics)
            self._print_dashboard(metrics)
            _log(f"Snapshot final → {_METRICS_PATH}")
            _log("Arrêté proprement.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NIDS Hybrid Orchestrator — pipeline metrics aggregator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--broker", type=str, default="localhost:29092", metavar="HOST:PORT",
        help="Kafka bootstrap server",
    )
    p.add_argument(
        "--interval", type=float, default=10.0, metavar="SECS",
        help="Seconds between dashboard refresh and JSON save",
    )
    p.add_argument(
        "--metrics-port", type=int, default=9100, metavar="PORT",
        help="Port for Prometheus /metrics endpoint",
    )
    p.add_argument(
        "--no-prometheus", action="store_true",
        help="Disable the Prometheus HTTP endpoint",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ColorFormatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    for _lib in ("kafka", "tensorflow", "keras", "absl"):
        logging.getLogger(_lib).setLevel(logging.WARNING)

    orch = HybridOrchestrator(
        kafka_broker=args.broker,
        save_interval=args.interval,
        metrics_port=None if args.no_prometheus else args.metrics_port,
    )
    orch.run()
