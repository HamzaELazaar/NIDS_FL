"""
hybrid/cloud_analyzer.py
--------------------------
Stage 2 — Deep cloud analysis with the XGBoost Optuna champion (7-class).

For each packet arriving on 'suspicious-traffic' (forwarded by the edge MLP):
  1. Decode raw packet_features + edge_prediction metadata.
  2. Scale features with the shared FeatureScaler.
  3. Classify with XGBoost → 7 classes + probability vector.
  4. Build an enriched alert with full pipeline provenance.
  5. Route:
       → 'classified-attacks'  always  (full audit trail)
       → 'alerts'              only if attack_type != 'Normal Traffic'
                               (false positives from edge stage are silently dropped)

Usage:
    python -m hybrid.cloud_analyzer
    python -m hybrid.cloud_analyzer --broker localhost:29092 --log-interval 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from federated_learning.preprocessing import FeatureScaler

# ── Topics ────────────────────────────────────────────────────────────────────

TOPIC_INPUT      = "suspicious-traffic"
TOPIC_CLASSIFIED = "classified-attacks"
TOPIC_ALERTS     = "alerts"

# ── Canonical paths ───────────────────────────────────────────────────────────

_FINAL         = _ROOT / "models" / "final"
_DEFAULT_MODEL = _FINAL / "best_centralized_multiclass.json"
_DEFAULT_ENC   = _FINAL / "label_encoder.joblib"
_DEFAULT_SCALER_DIR = _FINAL

# ── Feature order (must match training pipeline) ──────────────────────────────

FEATURE_ORDER: List[str] = [
    "Average Packet Size",
    "Subflow Fwd Bytes",
    "Destination Port",
    "Init_Win_bytes_forward",
    "PSH Flag Count",
    "FIN Flag Count",
    "ACK Flag Count",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Total Fwd Packets",
]

# ── Severity mapping ──────────────────────────────────────────────────────────
# Derived from attack impact and XGBoost per-class F1 scores.

_HIGH_SEVERITY   = {"DDoS", "DoS", "Bots"}
_MEDIUM_SEVERITY = {"Brute Force", "Web Attacks", "Port Scanning"}
_BENIGN_LABELS   = {"Normal Traffic", "BENIGN"}  # CICIDS2017 variant names

# ── Logging ───────────────────────────────────────────────────────────────────

_ANSI = {
    "DEBUG":   "\033[36m",
    "INFO":    "\033[31m",   # red for 🟥 stage 2
    "WARNING": "\033[33m",
    "ERROR":   "\033[31;1m",
    "RESET":   "\033[0m",
}


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _ANSI.get(record.levelname, _ANSI["RESET"])
        record.levelname = f"{color}{record.levelname:7}{_ANSI['RESET']}"
        return super().format(record)


logger = logging.getLogger(__name__)
_PREFIX = "🟥 [Cloud]"


def _log(msg: str, level: str = "info") -> None:
    getattr(logger, level)("%s %s", _PREFIX, msg)


# ─────────────────────────────────────────────────────────────────────────────
# Severity helper (module-level — reusable by tests)
# ─────────────────────────────────────────────────────────────────────────────

def compute_severity(attack_type: str, confidence: float) -> str:
    """Map an attack class + XGBoost confidence to a severity label.

    Rules (ordered by priority):
        high   — DDoS / DoS / Bots with confidence ≥ 0.65
        high   — any attack with confidence ≥ 0.90
        medium — Brute Force / Web Attacks / Port Scanning with confidence ≥ 0.55
        medium — any attack with confidence ≥ 0.65
        low    — confirmed attack below thresholds above

    Normal Traffic (edge false positive) returns "none" and is never alerted.
    """
    if attack_type in _BENIGN_LABELS:
        return "none"
    if attack_type in _HIGH_SEVERITY and confidence >= 0.65:
        return "high"
    if confidence >= 0.90:
        return "high"
    if attack_type in _MEDIUM_SEVERITY and confidence >= 0.55:
        return "medium"
    if confidence >= 0.65:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# CloudAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class CloudAnalyzer:
    """XGBoost Stage 2 classifier for the hybrid NIDS pipeline.

    Consumes traffic forwarded by the edge MLP (``suspicious-traffic``),
    performs 7-class attack classification, and publishes enriched alerts with
    full provenance (edge + cloud predictions, combined latency, severity).

    Args:
        model_path:   Path to the XGBoost model file (.json).
        encoder_path: Path to the fitted sklearn LabelEncoder (.joblib).
        kafka_broker: Kafka bootstrap address (host:port).
        log_interval: Log a summary line every N messages processed.
    """

    def __init__(
        self,
        model_path:   str = str(_DEFAULT_MODEL),
        encoder_path: str = str(_DEFAULT_ENC),
        kafka_broker: str = "localhost:29092",
        log_interval: int = 50,
    ) -> None:
        self.model_path   = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.kafka_broker = kafka_broker
        self.log_interval = log_interval

        self._booster: Any              = None
        self._le: Any                   = None
        self._scaler: Optional[FeatureScaler] = None
        self._producer: Any             = None
        self._consumer: Any             = None

        self._stats: Dict[str, int] = {
            "total":          0,
            "alerts":         0,
            "false_positives": 0,
            "parse_errors":   0,
        }
        # Per-class alert counters (populated at first classify call)
        self._class_counts: Dict[str, int] = {}

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the XGBoost Optuna champion and the LabelEncoder."""
        import xgboost as xgb  # type: ignore
        import joblib

        self._booster = xgb.Booster()
        self._booster.load_model(str(self.model_path))

        self._le = joblib.load(str(self.encoder_path))
        n_classes = len(self._le.classes_)

        _log(
            f"XGBoost chargé  model={self.model_path.name}  "
            f"n_classes={n_classes}  "
            f"classes={list(self._le.classes_)}"
        )

        # Initialise per-class counters
        self._class_counts = {c: 0 for c in self._le.classes_}

    def _load_scaler(self) -> None:
        """Load the fitted FeatureScaler (clip + StandardScale)."""
        import sys as _sys
        # The scaler was pickled under __main__ when preprocessing.py ran as
        # a script; inject the class so joblib can resolve it at load time.
        _sys.modules["__main__"].FeatureScaler = FeatureScaler  # type: ignore[attr-defined]
        self._scaler = FeatureScaler.load(str(_DEFAULT_SCALER_DIR))
        _log("Scaler chargé")

    def _init_kafka(self) -> None:
        from kafka import KafkaConsumer, KafkaProducer  # type: ignore

        self._producer = KafkaProducer(
            bootstrap_servers=[self.kafka_broker],
            acks=1,
            linger_ms=5,
            compression_type="gzip",
        )
        self._consumer = KafkaConsumer(
            TOPIC_INPUT,
            bootstrap_servers=[self.kafka_broker],
            group_id="cloud_analyzer",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            fetch_max_bytes=5 * 1024 * 1024,
        )
        self._consumer.poll(timeout_ms=300)
        _log(
            f"Kafka connecté  broker={self.kafka_broker}  "
            f"in={TOPIC_INPUT!r}  "
            f"out={TOPIC_CLASSIFIED!r},{TOPIC_ALERTS!r}"
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def classify(
        self,
        features: Dict[str, float] | np.ndarray,
    ) -> Dict[str, Any]:
        """Run XGBoost 7-class classification on a single packet.

        Args:
            features: Raw feature dict (keys = FEATURE_ORDER) OR pre-scaled
                      1-D numpy array of length 11.

        Returns:
            Dict with keys: attack_type (str), confidence (float),
                            top_3_classes (list), cloud_latency_ms (float).
        """
        import xgboost as xgb  # type: ignore

        if isinstance(features, dict):
            raw = np.array(
                [float(features[f]) for f in FEATURE_ORDER], dtype=np.float32
            ).reshape(1, -1)
            x = self._scaler.transform(raw)
        else:
            x = np.asarray(features, dtype=np.float32).reshape(1, -1)

        t0     = time.perf_counter()
        dmat   = xgb.DMatrix(x)
        probas = self._booster.predict(dmat)[0]     # shape (7,)
        lat_ms = (time.perf_counter() - t0) * 1_000

        top_idx = np.argsort(probas)[::-1]
        top_3 = [
            {
                "class": self._le.classes_[i],
                "prob":  round(float(probas[i]), 4),
            }
            for i in top_idx[:3]
        ]

        best    = int(top_idx[0])
        return {
            "attack_type":      self._le.classes_[best],
            "confidence":       round(float(probas[best]), 4),
            "top_3_classes":    top_3,
            "cloud_latency_ms": round(lat_ms, 2),
        }

    # ── Message handling ──────────────────────────────────────────────────────

    def _parse_message(self, msg: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Decode a Kafka message from the suspicious-traffic topic.

        Returns:
            (packet_features_dict, edge_prediction_dict)
        """
        payload: Dict[str, Any] = json.loads(msg.value.decode("utf-8"))
        packet_features = payload.get("packet_features", payload)
        edge_prediction = payload.get("edge_prediction", {})
        return packet_features, edge_prediction

    def _build_alert(
        self,
        packet_features: Dict[str, Any],
        edge_pred: Dict[str, Any],
        cloud_pred: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble the enriched alert document."""
        edge_lat   = float(edge_pred.get("edge_latency_ms", 0.0))
        cloud_lat  = cloud_pred["cloud_latency_ms"]
        attack_type = cloud_pred["attack_type"]
        confidence  = cloud_pred["confidence"]

        return {
            # Network identity (optional — present if upstream pipeline includes them)
            "src_ip":    packet_features.get("src_ip",   "unknown"),
            "dst_ip":    packet_features.get("dst_ip",   "unknown"),
            "src_port":  packet_features.get("src_port", "unknown"),
            "dst_port":  packet_features.get("dst_port", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),

            # Stage 1 context
            "edge_prediction": {
                "is_attack":  edge_pred.get("is_attack"),
                "confidence": edge_pred.get("confidence"),
                "latency_ms": edge_lat,
                "reason":     edge_pred.get("reason"),
                "model":      edge_pred.get("model_version", "fl_v3"),
            },

            # Stage 2 result
            "cloud_prediction": {
                "attack_type":     attack_type,
                "confidence":      confidence,
                "top_3":           cloud_pred["top_3_classes"],
                "cloud_latency_ms": cloud_lat,
                "model":           "xgboost_optuna_v1",
            },

            "pipeline_total_latency_ms": round(edge_lat + cloud_lat, 2),
            "severity": compute_severity(attack_type, confidence),
        }

    def _publish(self, alert: Dict[str, Any], is_attack: bool) -> None:
        """Publish the alert to the appropriate topics."""
        body = json.dumps(alert).encode("utf-8")
        headers = [
            ("attack_type", alert["cloud_prediction"]["attack_type"].encode()),
            ("severity",    alert["severity"].encode()),
        ]

        # Always record the full classification
        self._producer.send(TOPIC_CLASSIFIED, value=body, headers=headers)

        # Only forward confirmed attacks to the alerting bus
        if is_attack:
            self._producer.send(TOPIC_ALERTS, value=body, headers=headers)

    def _process_message(self, msg: Any) -> None:
        """Full pipeline for one incoming message: parse → classify → publish."""
        packet_features, edge_pred = self._parse_message(msg)
        cloud_pred  = self.classify(packet_features)
        attack_type = cloud_pred["attack_type"]

        alert     = self._build_alert(packet_features, edge_pred, cloud_pred)
        is_attack = attack_type not in _BENIGN_LABELS

        self._publish(alert, is_attack)

        # Counters
        self._stats["total"] += 1
        self._class_counts[attack_type] = self._class_counts.get(attack_type, 0) + 1
        if is_attack:
            self._stats["alerts"] += 1
        else:
            self._stats["false_positives"] += 1

        # Per-message debug log
        severity = alert["severity"]
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢", "none": "⚪"}.get(severity, "❔")
        logger.debug(
            "%s %s %-18s  conf=%.3f  severity=%s  lat=%.1fms",
            _PREFIX, icon, attack_type,
            cloud_pred["confidence"], severity,
            alert["pipeline_total_latency_ms"],
        )

        # Periodic summary
        if self._stats["total"] % self.log_interval == 0:
            self._print_stats()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _print_stats(self) -> None:
        total = self._stats["total"]
        fp    = self._stats["false_positives"]
        al    = self._stats["alerts"]
        fp_pct = 100 * fp / total if total else 0.0

        top_attacks = sorted(
            [(c, n) for c, n in self._class_counts.items() if c not in _BENIGN_LABELS and n > 0],
            key=lambda kv: -kv[1],
        )[:3]
        top_str = "  ".join(f"{c}={n}" for c, n in top_attacks) or "—"

        _log(
            f"total={total:,}  alerts={al:,}  "
            f"false_pos={fp:,} ({fp_pct:.1f}%)  "
            f"top_attacks: {top_str}"
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the cloud analyzer.  Blocks until interrupted (Ctrl-C)."""
        _log(
            f"Démarrage Stage 2  broker={self.kafka_broker}  "
            f"log_every={self.log_interval}"
        )

        self._load_model()
        self._load_scaler()
        self._init_kafka()

        _log(f"En écoute sur {TOPIC_INPUT!r} …")

        try:
            while True:
                records = self._consumer.poll(timeout_ms=100)

                for _tp, messages in records.items():
                    for msg in messages:
                        try:
                            self._process_message(msg)
                        except (KeyError, ValueError, json.JSONDecodeError) as exc:
                            self._stats["parse_errors"] += 1
                            _log(f"Parse error: {exc}", "warning")
                        except Exception as exc:
                            _log(f"Erreur inattendue: {exc}", "error")

        except KeyboardInterrupt:
            _log("Arrêt demandé (Ctrl-C)")
        finally:
            if self._producer:
                self._producer.flush()
                self._producer.close()
            if self._consumer:
                self._consumer.close()
            self._print_stats()
            _log("Arrêté proprement.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NIDS Cloud Analyzer — Stage 2 XGBoost 7-class classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--broker",       type=str, default="localhost:29092", metavar="HOST:PORT", help="Kafka bootstrap server")
    p.add_argument("--model",        type=str, default=str(_DEFAULT_MODEL), metavar="PATH",   help="XGBoost model file (.json)")
    p.add_argument("--encoder",      type=str, default=str(_DEFAULT_ENC),   metavar="PATH",   help="LabelEncoder file (.joblib)")
    p.add_argument("--log-interval", type=int, default=50,                  metavar="N",      help="Print stats every N messages")
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
    logging.getLogger("root").setLevel(logging.DEBUG)   # show per-packet debug
    for _lib in ("kafka", "tensorflow", "keras", "absl"):
        logging.getLogger(_lib).setLevel(logging.WARNING)

    analyzer = CloudAnalyzer(
        model_path=args.model,
        encoder_path=args.encoder,
        kafka_broker=args.broker,
        log_interval=args.log_interval,
    )
    analyzer.run()
