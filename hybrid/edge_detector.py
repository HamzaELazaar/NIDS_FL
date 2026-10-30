"""
hybrid/edge_detector.py
------------------------
Stage 1 — Lightweight edge screening with the federated binary MLP.

Routing logic (threshold = confidence gate, default 0.95):

    HIGH confidence  + class=Attack  → forward  (confident attack)
    HIGH confidence  + class=Normal  → DROP      (safe, never reaches stage 2)
    LOW  confidence  (uncertain)     → forward  (needs XGBoost / stage 2 analysis)

Batch inference (default batch_size=32) amortises Keras predict overhead
and dramatically increases throughput over per-packet calls.

Usage:
    python -m hybrid.edge_detector --threshold 0.95
    python -m hybrid.edge_detector --threshold 0.90 --batch-size 64 --broker localhost:29092
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Project root on sys.path (needed for federated_learning.preprocessing) ───
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from federated_learning.aggregation import (
    get_model_fingerprint,
    load_initial_weights,
    weights_size_kb,
)
from federated_learning.model import build_mlp, set_weights
from federated_learning.preprocessing import FeatureScaler

# ── Topics ────────────────────────────────────────────────────────────────────

TOPIC_INPUT  = "network-events"
TOPIC_OUTPUT = "suspicious-traffic"

# ── Model architecture (matches best_mlp_binary_weights_init.pkl) ─────────────

_FL_HIDDEN    = (256, 128, 64, 32)
_FL_INPUT_DIM = 11
_MODEL_VERSION = "fl_v3"

# Canonical paths
_FINAL = _ROOT / "models" / "final"
_DEFAULT_MODEL_PATH = _FINAL / "best_mlp_binary.keras"
_DEFAULT_PKL_PATH   = _FINAL / "best_mlp_binary_weights_init.pkl"
_DEFAULT_SCALER_DIR = _FINAL

# 11 features in the exact order expected by the scaler and model
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

# ── Logging ───────────────────────────────────────────────────────────────────

_ANSI = {
    "DEBUG":   "\033[36m",
    "INFO":    "\033[35m",   # magenta for 🟪 stage 1
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
_PREFIX = "🟪 [Edge]"


def _log(msg: str, level: str = "info") -> None:
    getattr(logger, level)("%s %s", _PREFIX, msg)


# ─────────────────────────────────────────────────────────────────────────────
# EdgeDetector
# ─────────────────────────────────────────────────────────────────────────────

class EdgeDetector:
    """Binary MLP edge filter — Stage 1 of the hybrid NIDS pipeline.

    Consumes raw network feature vectors from ``network-events``, runs fast
    binary inference (Normal vs Attack), and forwards uncertain or confirmed
    attack traffic to ``suspicious-traffic`` for deeper XGBoost analysis.

    Args:
        model_path:   Path to the binary MLP model (.keras or .pkl).
                      .keras files are resolved to a sibling .pkl automatically
                      because Lambda layers (focal loss) block direct Keras
                      deserialization.
        kafka_broker: Kafka bootstrap address (host:port).
        threshold:    Confidence gate in [0.5, 1.0].  A packet is routed based
                      on whether its confidence (= max(p, 1-p)) meets this bar.
        batch_size:   Maximum packets to accumulate before a single predict call.
    """

    def __init__(
        self,
        model_path: str = str(_DEFAULT_MODEL_PATH),
        kafka_broker: str = "localhost:29092",
        threshold: float = 0.95,
        batch_size: int = 32,
    ) -> None:
        if not 0.5 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.5, 1.0], got {threshold}")

        self.model_path   = Path(model_path)
        self.kafka_broker = kafka_broker
        self.threshold    = threshold
        self.batch_size   = batch_size

        self._model: Any              = None
        self._scaler: FeatureScaler   = None
        self._fingerprint: str        = ""
        self._producer: Any           = None
        self._consumer: Any           = None

        # Counters — reset each stats interval
        self._stats: Dict[str, int] = {
            "total":         0,
            "forwarded":     0,
            "dropped":       0,
            "uncertain":     0,
            "parse_errors":  0,
        }
        self._stats_start = time.time()
        self._stats_interval = 10.0     # print throughput every N seconds
        self._batch_timeout  = 0.1      # flush partial batch after N seconds

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _resolve_pkl(self) -> Path:
        """Derive the .pkl weights path from a .keras model_path, or use as-is."""
        if self.model_path.suffix == ".pkl":
            return self.model_path

        # .keras → look for sibling *_weights_init.pkl
        candidate = self.model_path.parent / (self.model_path.stem + "_weights_init.pkl")
        if candidate.exists():
            return candidate

        # Fallback to the canonical pkl
        return _DEFAULT_PKL_PATH

    def _load_model(self) -> None:
        """Load the binary MLP.

        If model_path is a .keras file that exists, load it directly with
        keras.models.load_model() (works for checkpoint models without Lambda
        layers).  Otherwise fall back to building the architecture from
        _FL_HIDDEN and warm-starting from the sibling .pkl weights file.
        """
        if self.model_path.suffix == ".keras" and self.model_path.exists():
            import keras  # type: ignore
            try:
                keras.config.enable_unsafe_deserialization()
                self._model = keras.models.load_model(str(self.model_path))
                self._fingerprint = "keras_direct"
                total_params = sum(int(w.numpy().size) for w in self._model.weights)
                _log(
                    f"Modèle chargé (keras)  "
                    f"path={self.model_path.name}  "
                    f"params={total_params:,}  "
                    f"fingerprint={self._fingerprint}"
                )
                return
            except Exception as exc:
                # Lambda layers used for focal loss block direct load — fall through
                # to pkl path which rebuilds clean inference architecture.
                _log(f"keras load skipped (Lambda layer) — using pkl fallback: {exc}")

        # Fallback: build architecture + load pkl weights
        self._model = build_mlp(
            input_dim=_FL_INPUT_DIM,
            n_classes=2,
            hidden_layers=_FL_HIDDEN,
            all_batchnorm=True,
        )
        pkl_path = self._resolve_pkl()
        init_w   = load_initial_weights(str(pkl_path))
        set_weights(self._model, init_w)
        self._fingerprint = get_model_fingerprint(init_w)
        _log(
            f"Modèle chargé (pkl)  "
            f"arch={_FL_HIDDEN}  "
            f"params={sum(w.size for w in init_w):,}  "
            f"size={weights_size_kb(init_w):.1f} KB  "
            f"fingerprint={self._fingerprint}"
        )

    def _load_scaler(self) -> None:
        """Load the fitted FeatureScaler (clip + StandardScale).

        The scaler was pickled when preprocessing.py ran as __main__, so joblib
        looks for __main__.FeatureScaler at load time.  Injecting the class into
        __main__ here resolves the AttributeError without modifying the pkl file.
        """
        import sys as _sys
        _sys.modules["__main__"].FeatureScaler = FeatureScaler  # type: ignore[attr-defined]
        self._scaler = FeatureScaler.load(str(_DEFAULT_SCALER_DIR))
        _log("Scaler chargé")

    def _warmup(self) -> None:
        """Run a dummy batch to trigger XLA/JIT compilation before live traffic."""
        dummy = np.zeros((self.batch_size, _FL_INPUT_DIM), dtype=np.float32)
        self._model.predict(dummy, verbose=0)
        _log(f"Warm-up OK  (batch_size={self.batch_size})")

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
            group_id="edge_detector",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            fetch_max_bytes=5 * 1024 * 1024,
        )
        self._consumer.poll(timeout_ms=300)     # lock in 'latest' offset
        _log(
            f"Kafka connecté  broker={self.kafka_broker}  "
            f"in={TOPIC_INPUT!r}  out={TOPIC_OUTPUT!r}"
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_packet(self, features: Dict[str, float] | np.ndarray) -> Dict[str, Any]:
        """Classify a single packet (public API).

        Args:
            features: Either a dict mapping feature name → float value,
                      or a pre-scaled 1-D numpy array of length 11.

        Returns:
            Dict with keys: is_suspicious, is_attack, confidence,
                            prediction (str), raw_proba.
        """
        if isinstance(features, dict):
            raw = np.array(
                [features[f] for f in FEATURE_ORDER], dtype=np.float32
            ).reshape(1, -1)
            x = self._scaler.transform(raw)
        else:
            x = np.asarray(features, dtype=np.float32).reshape(1, -1)

        t0    = time.perf_counter()
        proba = float(self._model.predict(x, verbose=0)[0, 0])
        lat   = (time.perf_counter() - t0) * 1_000

        return self._build_prediction(proba, lat)

    def _predict_batch(
        self,
        features_batch: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Run inference on a (N, 11) scaled array.

        Returns:
            (predictions list, per-packet latency in ms)
        """
        t0     = time.perf_counter()
        probas = self._model.predict(features_batch, verbose=0)[:, 0]
        lat_ms = (time.perf_counter() - t0) * 1_000 / len(probas)
        return [self._build_prediction(float(p), lat_ms) for p in probas], lat_ms

    def _build_prediction(self, proba: float, lat_ms: float) -> Dict[str, Any]:
        """Derive routing decision from a raw sigmoid probability.

        Routing:
            confidence ≥ threshold AND is_attack  → forward  (certain attack)
            confidence ≥ threshold AND NOT is_attack → DROP  (certain normal)
            confidence <  threshold                → forward  (uncertain → stage 2)
        """
        is_attack  = proba >= 0.5
        confidence = proba if is_attack else (1.0 - proba)
        is_certain = confidence >= self.threshold

        if is_certain and is_attack:
            reason       = "attack_confident"
            is_suspicious = True
        elif is_certain:                         # certain Normal
            reason       = "normal_dropped"
            is_suspicious = False
        else:
            reason       = "uncertain"
            is_suspicious = True

        return {
            "is_suspicious":   is_suspicious,
            "is_attack":       bool(is_attack),
            "confidence":      round(confidence, 4),
            "prediction":      "Attack" if is_attack else "Normal",
            "raw_proba":       round(proba, 4),
            "reason":          reason,
            "edge_latency_ms": round(lat_ms, 2),
        }

    # ── Message handling ──────────────────────────────────────────────────────

    def _parse_message(
        self,
        msg: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode a Kafka message into a scaled feature array.

        Expects the message value to be JSON with the 11 CICIDS2017 feature keys.

        Returns:
            (scaled_features_1d, original_dict)
        """
        payload: Dict[str, Any] = json.loads(msg.value.decode("utf-8"))
        raw = np.array(
            [float(payload[f]) for f in FEATURE_ORDER], dtype=np.float32
        ).reshape(1, -1)
        scaled = self._scaler.transform(raw)[0]
        return scaled, payload

    def _process_batch(
        self,
        buffer: List[Tuple[Any, np.ndarray, Dict[str, Any]]],
    ) -> None:
        """Run batch inference and route each packet.

        Args:
            buffer: List of (raw_msg, scaled_features_1d, original_dict).
        """
        features_matrix = np.stack([f for _, f, _ in buffer])
        predictions, lat_ms = self._predict_batch(features_matrix)

        for (msg, _, original), pred in zip(buffer, predictions):
            self._stats["total"] += 1

            output: Dict[str, Any] = {
                "packet_features": original,
                "edge_prediction": {
                    "is_attack":       pred["is_attack"],
                    "confidence":      pred["confidence"],
                    "model_version":   _MODEL_VERSION,
                    "edge_latency_ms": pred["edge_latency_ms"],
                    "reason":          pred["reason"],
                    "fingerprint":     self._fingerprint,
                },
            }

            if pred["is_suspicious"]:
                self._producer.send(
                    TOPIC_OUTPUT,
                    value=json.dumps(output).encode("utf-8"),
                    headers=[
                        ("edge_decision", pred["reason"].encode()),
                        ("is_attack",     str(pred["is_attack"]).encode()),
                    ],
                )
                self._stats["forwarded"] += 1
                if pred["reason"] == "uncertain":
                    self._stats["uncertain"] += 1
            else:
                self._stats["dropped"] += 1

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _print_stats(self) -> None:
        """Log a one-line throughput and routing summary."""
        elapsed = time.time() - self._stats_start
        total   = self._stats["total"]
        pps     = total / elapsed if elapsed > 0 else 0.0
        fwd_pct = 100 * self._stats["forwarded"] / total if total else 0.0
        drp_pct = 100 * self._stats["dropped"]   / total if total else 0.0
        unc_pct = 100 * self._stats["uncertain"]  / total if total else 0.0

        _log(
            f"packets/s={pps:>8,.0f}  "
            f"forwarded={fwd_pct:5.1f}%  "
            f"dropped={drp_pct:5.1f}%  "
            f"uncertain={unc_pct:5.1f}%  "
            f"total={total:,}  "
            f"errors={self._stats['parse_errors']}"
        )

        # Reset window
        self._stats = {k: 0 for k in self._stats}
        self._stats_start = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the edge detector.  Blocks until interrupted (Ctrl-C)."""
        _log(
            f"Démarrage  threshold={self.threshold}  "
            f"batch_size={self.batch_size}  "
            f"broker={self.kafka_broker}  "
            f"version={_MODEL_VERSION}"
        )

        self._load_model()
        self._load_scaler()
        self._init_kafka()
        self._warmup()

        _log(f"En écoute sur {TOPIC_INPUT!r} …")

        buffer: List[Tuple[Any, np.ndarray, Dict[str, Any]]] = []
        last_flush_time = time.time()
        last_stats_time = time.time()

        try:
            while True:
                # ── Poll (50 ms burst window) ────────────────────────────────
                records = self._consumer.poll(timeout_ms=50)

                for _tp, messages in records.items():
                    for msg in messages:
                        try:
                            scaled, original = self._parse_message(msg)
                            buffer.append((msg, scaled, original))
                        except (KeyError, ValueError, json.JSONDecodeError) as exc:
                            self._stats["parse_errors"] += 1
                            logger.debug("%s Parse error: %s", _PREFIX, exc)

                # ── Flush when batch is full OR timeout elapsed ───────────────
                now = time.time()
                should_flush = (
                    len(buffer) >= self.batch_size
                    or (buffer and (now - last_flush_time) >= self._batch_timeout)
                )
                if should_flush:
                    self._process_batch(buffer)
                    buffer.clear()
                    last_flush_time = now

                # ── Stats every stats_interval seconds ───────────────────────
                if now - last_stats_time >= self._stats_interval:
                    self._print_stats()
                    last_stats_time = now

        except KeyboardInterrupt:
            _log("Arrêt demandé (Ctrl-C)")
        finally:
            if buffer:
                self._process_batch(buffer)
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
        description="NIDS Edge Detector — Stage 1 binary MLP filter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--threshold", type=float, default=0.95, metavar="T",
        help="Confidence gate: packets above T are routed with certainty",
    )
    p.add_argument(
        "--batch-size", type=int, default=32, metavar="N",
        help="Maximum packets per inference call",
    )
    p.add_argument(
        "--broker", type=str, default="localhost:29092", metavar="HOST:PORT",
        help="Kafka bootstrap server",
    )
    p.add_argument(
        "--model", type=str, default=str(_DEFAULT_MODEL_PATH), metavar="PATH",
        help="Path to .keras model or .pkl weights file",
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

    det = EdgeDetector(
        model_path=args.model,
        kafka_broker=args.broker,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    det.run()
