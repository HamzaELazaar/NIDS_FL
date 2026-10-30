"""
federated_learning/fl_server.py
---------------------------------
FedAvg aggregation server for NIDS binary FL.

Each round:
  1. Collect weight messages from n_clients on 'model-weights'.
  2. Aggregate with FedAvg (federated_average).
  3. Evaluate the global model on the held-out test set.
  4. Broadcast the result on 'global-model'.
  5. Log metrics and persist history.

Usage:
    python -m federated_learning.fl_server --rounds 10 --clients 3
    python -m federated_learning.fl_server --rounds 20 --clients 3 --timeout 180
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

from federated_learning.aggregation import (
    compute_weight_delta,
    deserialize_weights,
    federated_average,
    get_model_fingerprint,
    load_initial_weights,
    serialize_weights,
    weights_size_kb,
)
from federated_learning.model import (
    build_mlp,
    evaluate_mlp,
    get_weights,
    set_weights,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT     = Path(__file__).parent.parent
_DATA     = _ROOT / "data" / "processed"
_MODELS   = _ROOT / "models"
_INIT_PKL = _ROOT / "models" / "final" / "best_mlp_binary_weights_init.pkl"
_HISTORY  = _ROOT / "data" / "fl_history.json"

# ── Kafka topics ──────────────────────────────────────────────────────────────

TOPIC_WEIGHTS  = "model-weights"
TOPIC_GLOBAL   = "global-model"
TOPIC_FL_ROUNDS = "fl-rounds"

# ── FL model architecture (matches best_mlp_binary_weights_init.pkl) ──────────

FL_HIDDEN    = (256, 128, 64, 32)
FL_INPUT_DIM = 11

# ── Logging ───────────────────────────────────────────────────────────────────

_ANSI = {
    "DEBUG":   "\033[36m",
    "INFO":    "\033[32m",
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
_PREFIX = "🟢 [Serveur]"


def _log(msg: str, level: str = "info") -> None:
    getattr(logger, level)("%s %s", _PREFIX, msg)


# ─────────────────────────────────────────────────────────────────────────────
# FLServer
# ─────────────────────────────────────────────────────────────────────────────

class FLServer:
    """Federated Learning aggregation server.

    Args:
        n_clients:    Number of clients expected each round.
        kafka_broker: Kafka bootstrap address (host:port).
        n_rounds:     Total FL communication rounds.
        timeout:      Seconds to wait for client weights before timing out.
    """

    def __init__(
        self,
        n_clients: int = 3,
        kafka_broker: str = "localhost:29092",
        n_rounds: int = 10,
        timeout: int = 120,
    ) -> None:
        self.n_clients    = n_clients
        self.kafka_broker = kafka_broker
        self.n_rounds     = n_rounds
        self.timeout      = timeout

        # Global model — built once, weights updated every round
        self.model: Any = None

        # Test set (loaded once at startup)
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        # Kafka handles
        self._producer: Any = None
        self._consumer: Any = None

        # State tracking
        self.history:     List[Dict[str, Any]] = []
        self._best_f1:    float = 0.0
        self._best_round: int   = 0
        self._prev_weights: Optional[List[np.ndarray]] = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _load_test_set(self) -> None:
        path = _DATA / "test.npz"
        data = np.load(path)
        self.X_test = data["X"].astype(np.float32)
        self.y_test = data["y_binary"].astype(np.int32)
        n0 = int((self.y_test == 0).sum())
        n1 = int((self.y_test == 1).sum())
        _log(
            f"Test set chargé : {len(self.X_test):,} samples  "
            f"(Normal={n0:,} / Attack={n1:,})"
        )

    def _build_model(self) -> None:
        """Build the global MLP and warm-start from centralized binary weights."""
        self.model = build_mlp(
            input_dim=FL_INPUT_DIM,
            n_classes=2,
            hidden_layers=FL_HIDDEN,
            learning_rate=1e-3,
            all_batchnorm=True,
        )
        try:
            init_w = load_initial_weights(str(_INIT_PKL))
            set_weights(self.model, init_w)
            _log(
                f"Warm-start OK — fingerprint={get_model_fingerprint(init_w)}  "
                f"({weights_size_kb(init_w):.1f} KB)"
            )
            self._prev_weights = init_w
        except Exception as exc:
            _log(f"Warm-start échoué ({exc}) — init aléatoire", "warning")
            self._prev_weights = get_weights(self.model)

    def _init_kafka(self) -> None:
        from kafka import KafkaConsumer, KafkaProducer  # type: ignore

        self._producer = KafkaProducer(
            bootstrap_servers=[self.kafka_broker],
            max_request_size=10 * 1024 * 1024,
            acks="all",
            retries=3,
        )
        # Server owns a dedicated consumer group for the weights topic.
        self._consumer = KafkaConsumer(
            TOPIC_WEIGHTS,
            bootstrap_servers=[self.kafka_broker],
            group_id="fl_server",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            fetch_max_bytes=10 * 1024 * 1024,
            max_partition_fetch_bytes=10 * 1024 * 1024,
        )
        # Force partition assignment now so 'latest' offset is locked before
        # round 1 starts — prevents missing the first batch of client messages.
        self._consumer.poll(timeout_ms=500)
        _log(f"Kafka connecté — broker={self.kafka_broker}")

    # ── Core methods ──────────────────────────────────────────────────────────

    def collect_weights(
        self,
        round_idx: int,
        timeout: Optional[int] = None,
    ) -> List[Tuple[int, List[np.ndarray], int]]:
        """Wait for weight messages from all clients for this round.

        Polls the 'model-weights' topic, filtering by the ``round_idx`` Kafka
        header. Stops early when all ``n_clients`` replies arrive. On timeout,
        returns whatever was collected (may be fewer than n_clients).

        Args:
            round_idx: Current FL round (matched against message headers).
            timeout:   Override instance timeout (seconds).

        Returns:
            List of ``(client_id, weights, n_samples)`` sorted by client_id.
        """
        tmo      = timeout if timeout is not None else self.timeout
        received: Dict[int, Tuple[List[np.ndarray], int]] = {}
        deadline = time.time() + tmo

        _log(
            f"Collecte — round={round_idx}  "
            f"attente de {self.n_clients} clients  "
            f"(timeout={tmo}s)"
        )

        while time.time() < deadline:
            if len(received) >= self.n_clients:
                break

            remaining_ms = max(1, int((deadline - time.time()) * 1000))
            records = self._consumer.poll(timeout_ms=min(1_000, remaining_ms))

            for _tp, messages in records.items():
                for msg in messages:
                    headers = {k: v.decode() for k, v in (msg.headers or [])}

                    msg_round = int(headers.get("round_idx", -1))
                    if msg_round != round_idx:
                        continue  # wrong round — skip silently

                    client_id = int(headers.get("client_id", -1))
                    n_samples = int(headers.get("n_samples",  0))

                    if client_id in received:
                        _log(f"Doublon client {client_id} ignoré", "warning")
                        continue

                    try:
                        weights, _ = deserialize_weights(msg.value)
                        received[client_id] = (weights, n_samples)
                        _log(
                            f"  Client {client_id} ✓  "
                            f"n={n_samples:,}  "
                            f"size={len(msg.value)/1024:.1f} KB  "
                            f"[{len(received)}/{self.n_clients}]"
                        )
                    except Exception as exc:
                        _log(
                            f"Erreur déserialisation client {client_id}: {exc}",
                            "error",
                        )

        if len(received) < self.n_clients:
            _log(
                f"Timeout — {len(received)}/{self.n_clients} clients reçus "
                f"pour round={round_idx}",
                "warning",
            )

        return [(cid, w, n) for cid, (w, n) in sorted(received.items())]

    def aggregate(
        self,
        client_data: List[Tuple[int, List[np.ndarray], int]],
    ) -> List[np.ndarray]:
        """Run FedAvg over the collected client weights.

        Args:
            client_data: Output of collect_weights.

        Returns:
            Aggregated global weight list (float32, same shape as client weights).
        """
        client_weights = [w   for _, w, _ in client_data]
        client_sizes   = [n   for _, _, n in client_data]
        ids            = [cid for cid, _, _ in client_data]
        total          = sum(client_sizes)

        _log(
            f"FedAvg — clients={ids}  "
            f"sizes={[f'{n:,}' for n in client_sizes]}  "
            f"N_total={total:,}"
        )
        return federated_average(client_weights, client_sizes, min_clients=2)

    def evaluate_global(self) -> Dict[str, Any]:
        """Evaluate the current global model on the held-out test set.

        Returns:
            Metrics dict with accuracy, f1_macro, f1_weighted, auc,
            precision_macro, recall_macro, confusion_matrix.
        """
        metrics = evaluate_mlp(self.model, self.X_test, self.y_test, n_classes=2)
        _log(
            f"Global test  "
            f"acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1_macro']:.4f}  "
            f"auc={metrics['auc']:.4f}  "
            f"prec={metrics['precision_macro']:.4f}  "
            f"rec={metrics['recall_macro']:.4f}"
        )
        return metrics

    def publish_global_model(
        self,
        weights: List[np.ndarray],
        round_idx: int,
    ) -> None:
        """Broadcast aggregated weights on the 'global-model' topic.

        Each client has its own consumer group, so every subscribed client
        independently receives this same message.

        Args:
            weights:   Aggregated global weights from FedAvg.
            round_idx: Round number (embedded in Kafka headers for filtering).
        """
        payload = serialize_weights(
            weights,
            metadata={"source": "server", "round": round_idx},
            compress=True,
        )
        headers = [
            (b"round_idx", str(round_idx).encode()),
            (b"source",    b"server"),
        ]
        future = self._producer.send(
            TOPIC_GLOBAL,
            value=payload,
            headers=headers,
            key=b"global",
        )
        self._producer.flush()
        record = future.get(timeout=10)

        _log(
            f"Modèle global publié  round={round_idx}  "
            f"size={len(payload)/1024:.1f} KB  "
            f"fingerprint={get_model_fingerprint(weights)}  "
            f"offset={record.offset}"
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _publish_round_metrics(self, record: dict) -> None:
        """Publish round metrics to fl-rounds Kafka topic for the dashboard."""
        gm = record.get("global_metrics", {})
        payload = {
            "round":              record["round"],
            "global_f1":          round(gm.get("f1_macro", 0.0), 4),
            "global_accuracy":    round(gm.get("accuracy", 0.0), 4),
            "global_loss":        round(gm.get("loss", 0.0), 4),
            "aggregation_method": "FedAvg",
            "timestamp":          record.get("timestamp", ""),
            "clients":            record.get("clients", []),
        }
        try:
            self._producer.send(
                TOPIC_FL_ROUNDS,
                value=json.dumps(payload, default=str).encode("utf-8"),
                key=str(record["round"]).encode(),
            )
            self._producer.flush()
        except Exception as exc:
            _log(f"fl-rounds publish failed: {exc}", "warning")

    def _save_history(self) -> None:
        with open(_HISTORY, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def _save_best(self) -> None:
        path = _MODELS / "fl_global_best.keras"
        self.model.save(str(path))
        _log(f"Meilleur modèle → {path.name}  (f1={self._best_f1:.4f}  round={self._best_round})")

    def _finalize(self) -> None:
        """Save the final global model, print summary, close Kafka handles."""
        final_path = _MODELS / "fl_global.keras"
        self.model.save(str(final_path))
        _log(f"Modèle final → {final_path.name}")

        last = self.history[-1] if self.history else {}
        gm   = last.get("global_metrics", {})
        _log(
            f"Résumé  rounds_exécutés={len(self.history)}  "
            f"best_round={self._best_round}  "
            f"best_f1={self._best_f1:.4f}  "
            f"final_f1={gm.get('f1_macro', 0):.4f}  "
            f"final_auc={gm.get('auc', 0):.4f}"
        )
        _log(f"Historique → {_HISTORY.relative_to(_ROOT)}")

        if self._producer:
            self._producer.close()
        if self._consumer:
            self._consumer.close()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, repeat_hours: float = 0.0) -> None:
        """Execute n_rounds of federated averaging.

        Args:
            repeat_hours: If > 0, restart a new training session after this
                          many hours (continuous mode).  Rounds are numbered
                          cumulatively across sessions so the convergence chart
                          shows the full lifetime history.
        """
        _log(
            f"Démarrage FedAvg  rounds={self.n_rounds}  "
            f"n_clients={self.n_clients}  "
            f"timeout={self.timeout}s  broker={self.kafka_broker}"
            + (f"  repeat={repeat_hours}h" if repeat_hours > 0 else "")
        )

        self._load_test_set()
        self._build_model()
        self._init_kafka()

        session = 0
        while True:
            session += 1
            round_offset = len(self.history)
            if session > 1:
                _log(f"{'═'*12} Session {session} — rounds {round_offset+1}…{round_offset+self.n_rounds} {'═'*12}")

            for round_idx in range(1, self.n_rounds + 1):
                cumulative_idx = round_offset + round_idx
                _log(f"{'═'*12} Round {cumulative_idx} (session {session}, local {round_idx}/{self.n_rounds}) {'═'*12}")
                t_start = time.time()

                # ── 1. Collect ────────────────────────────────────────────────
                client_data = self.collect_weights(round_idx)
                n_received  = len(client_data)

                if n_received < 2:
                    _log(
                        f"Round {cumulative_idx} ignoré — {n_received} client(s) reçu(s) "
                        "(minimum 2 requis)",
                        "warning",
                    )
                    continue

                # ── 2. Aggregate ──────────────────────────────────────────────
                new_weights = self.aggregate(client_data)

                # ── 3. Weight delta (convergence monitor) ─────────────────────
                delta     = compute_weight_delta(self._prev_weights, new_weights)
                converged = delta["converged"]

                # ── 4. Update global model ────────────────────────────────────
                set_weights(self.model, new_weights)
                self._prev_weights = new_weights

                # ── 5. Evaluate on test set ───────────────────────────────────
                global_metrics = self.evaluate_global()
                f1  = global_metrics["f1_macro"]
                auc = global_metrics["auc"]

                # ── 6. Save best ──────────────────────────────────────────────
                if f1 > self._best_f1:
                    self._best_f1    = f1
                    self._best_round = cumulative_idx
                    self._save_best()

                # ── 7. Publish to all clients ─────────────────────────────────
                self.publish_global_model(new_weights, round_idx)

                # ── 8. Record and persist ─────────────────────────────────────
                latency_ms = int((time.time() - t_start) * 1000)
                self.history.append({
                    "round":              cumulative_idx,
                    "session":            session,
                    "n_clients_received": n_received,
                    "global_metrics":     global_metrics,
                    "weight_delta":       delta,
                    "latency_ms":         latency_ms,
                    "timestamp":          datetime.now(timezone.utc).isoformat(),
                })
                self._save_history()
                self._publish_round_metrics(self.history[-1])

                # ── 9. Round summary ──────────────────────────────────────────
                tags = ""
                if converged:
                    tags += "  ✅ CONVERGÉ"
                if cumulative_idx == self._best_round:
                    tags += "  ★ MEILLEUR"
                _log(
                    f"Round {cumulative_idx}  "
                    f"clients={n_received}/{self.n_clients}  "
                    f"f1={f1:.4f}  "
                    f"auc={auc:.4f}  "
                    f"Δ={delta['mean_delta']:.2e}  "
                    f"{latency_ms}ms"
                    f"{tags}"
                )

            _log("═" * 40)
            self._finalize()

            if repeat_hours <= 0:
                break

            next_run = datetime.now(timezone.utc).replace(microsecond=0)
            sleep_s  = int(repeat_hours * 3600)
            _log(
                f"Session {session} terminée — prochaine session dans "
                f"{repeat_hours:.1f}h  (≈ {sleep_s//3600}h{(sleep_s%3600)//60:02d}m)"
            )
            time.sleep(sleep_s)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NIDS Federated Learning server (FedAvg)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rounds",        type=int,   default=10,                metavar="N",         help="FL rounds per session")
    p.add_argument("--clients",       type=int,   default=3,                 metavar="N",         help="Expected clients per round")
    p.add_argument("--broker",        type=str,   default="localhost:29092", metavar="HOST:PORT", help="Kafka bootstrap server")
    p.add_argument("--timeout",       type=int,   default=120,               metavar="S",         help="Seconds to wait for client weights per round")
    p.add_argument("--repeat-hours",  type=float, default=0.0,               metavar="H",         help="Repeat training every H hours (0 = run once). Use 24 for daily, 168 for weekly.")
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

    server = FLServer(
        n_clients=args.clients,
        kafka_broker=args.broker,
        n_rounds=args.rounds,
        timeout=args.timeout,
    )
    server.run(repeat_hours=args.repeat_hours)
