"""
federated_learning/fl_client.py
---------------------------------
Federated Learning client for NIDS binary classification (Normal vs Attack).

Each client:
  1. Loads its local data partition.
  2. Trains the MLP locally for local_epochs.
  3. Evaluates on its local validation set.
  4. Publishes weights to Kafka topic 'model-weights'.
  5. Waits for the aggregated global model on 'global-model'.
  6. Updates local weights and repeats for n_rounds.

Usage:
    python -m federated_learning.fl_client --id 1 --rounds 10 --epochs 5
    python -m federated_learning.fl_client --id 2 --broker localhost:29092
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from federated_learning.aggregation import (
    deserialize_weights,
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

_ROOT      = Path(__file__).parent.parent
_DATA_DIR  = _ROOT / "data" / "processed"
_MODELS    = _ROOT / "models"
_INIT_PKL  = _ROOT / "models" / "final" / "best_mlp_binary_weights_init.pkl"

# ── Kafka topics ──────────────────────────────────────────────────────────────

TOPIC_WEIGHTS = "model-weights"
TOPIC_GLOBAL  = "global-model"

# ── FL model architecture (must match best_mlp_binary_weights_init.pkl) ───────
# 26 tensors: Dense(11→256→128→64→32→1) with BatchNorm on all hidden layers.

FL_HIDDEN    = (256, 128, 64, 32)
FL_INPUT_DIM = 11

# ── Emoji per client ID ───────────────────────────────────────────────────────

_EMOJI: Dict[int, str] = {1: "🟦", 2: "🟪", 3: "🟧"}


# ── Colored logging ───────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# FLClient
# ─────────────────────────────────────────────────────────────────────────────

class FLClient:
    """One federated learning participant.

    Args:
        client_id:    Integer identifier (1, 2, or 3).
        kafka_broker: Kafka bootstrap address (host:port).
        n_rounds:     Number of FL communication rounds to execute.
        local_epochs: Keras epochs per local training step.
    """

    def __init__(
        self,
        client_id: int,
        kafka_broker: str = "localhost:29092",
        n_rounds: int = 10,
        local_epochs: int = 5,
    ) -> None:
        self.client_id    = client_id
        self.kafka_broker = kafka_broker
        self.n_rounds     = n_rounds
        self.local_epochs = local_epochs
        self.emoji        = _EMOJI.get(client_id, "🔵")
        self._prefix      = f"{self.emoji} [Client {client_id}]"

        # Populated by load_local_data
        self.X_train:  Optional[np.ndarray] = None
        self.y_train:  Optional[np.ndarray] = None
        self.X_val:    Optional[np.ndarray] = None
        self.y_val:    Optional[np.ndarray] = None
        self.n_samples: int = 0

        # Keras model — built once, weights updated every round
        self.model: Any = None

        # Kafka (created in run())
        self._producer: Any = None
        self._consumer: Any = None

        # Per-round training history
        self.history: list = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "info") -> None:
        getattr(logger, level)("%s %s", self._prefix, msg)

    # ── Data ──────────────────────────────────────────────────────────────────

    def load_local_data(self) -> None:
        """Load the client's local data partition.

        Reads ``data/processed/fl_client_{id}.npz`` (keys: X, y_binary).
        Falls back to partitioning ``train.npz`` into 3 equal shards when the
        per-client file does not yet exist, and saves the result for future runs.
        """
        client_path = _DATA_DIR / f"fl_client_{self.client_id}.npz"

        if client_path.exists():
            data = np.load(client_path)
            X = data["X"].astype(np.float32)
            y = data["y_binary"].astype(np.int32)
            self._log(f"Données chargées : {client_path.name}  ({len(X):,} samples)")
        else:
            self._log(
                f"{client_path.name} absent — partition automatique depuis train.npz",
                "warning",
            )
            X, y = self._partition_from_train()

        # Stratified 80/20 train/val split seeded per client
        rng   = np.random.default_rng(seed=self.client_id * 42)
        idx   = rng.permutation(len(X))
        split = int(0.8 * len(X))
        t_idx, v_idx = idx[:split], idx[split:]

        self.X_train  = X[t_idx]
        self.y_train  = y[t_idx]
        self.X_val    = X[v_idx]
        self.y_val    = y[v_idx]
        self.n_samples = len(self.X_train)

        n_normal = int((self.y_train == 0).sum())
        n_attack = int((self.y_train == 1).sum())
        self._log(
            f"Train={self.n_samples:,}  (Normal={n_normal:,} / Attack={n_attack:,})  "
            f"Val={len(self.X_val):,}"
        )

    def _partition_from_train(self) -> tuple[np.ndarray, np.ndarray]:
        """Partition train.npz into 3 equal shards and persist the result."""
        train_path = _DATA_DIR / "train.npz"
        if not train_path.exists():
            raise FileNotFoundError(f"train.npz not found at {train_path}")

        data = np.load(train_path)
        X = data["X"].astype(np.float32)
        y = data["y_binary"].astype(np.int32)

        rng        = np.random.default_rng(seed=0)
        idx        = rng.permutation(len(X))
        shard_size = len(X) // 3
        c          = self.client_id - 1          # 1-indexed → 0-indexed
        start      = c * shard_size
        end        = start + shard_size if self.client_id < 3 else len(X)
        shard      = idx[start:end]

        client_path = _DATA_DIR / f"fl_client_{self.client_id}.npz"
        np.savez_compressed(client_path, X=X[shard], y_binary=y[shard])
        self._log(f"Partition sauvegardée → {client_path.name}  ({len(shard):,} samples)")

        return X[shard], y[shard]

    # ── Model ─────────────────────────────────────────────────────────────────

    def _build_model(self) -> None:
        """Build the binary MLP and warm-start from centralized weights."""
        self.model = build_mlp(
            input_dim=FL_INPUT_DIM,
            n_classes=2,
            hidden_layers=FL_HIDDEN,
            learning_rate=1e-3,
            all_batchnorm=True,   # required to match 26-tensor pkl architecture
        )
        try:
            init_w = load_initial_weights(str(_INIT_PKL))
            set_weights(self.model, init_w)
            self._log(
                f"Warm-start OK — fingerprint={get_model_fingerprint(init_w)}  "
                f"({weights_size_kb(init_w):.1f} KB)"
            )
        except Exception as exc:
            self._log(f"Warm-start échoué ({exc}) — init aléatoire", "warning")

    # ── Local training ────────────────────────────────────────────────────────

    def train_local(self) -> Dict[str, Any]:
        """Train the model locally for ``local_epochs`` epochs.

        Returns:
            Dict with keys: loss, accuracy, n_samples, train_time_s.
        """
        classes = np.unique(self.y_train)
        cw_arr  = compute_class_weight("balanced", classes=classes, y=self.y_train)
        class_weight = dict(zip(classes.tolist(), cw_arr.tolist()))

        t0   = time.time()
        hist = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.local_epochs,
            batch_size=256,
            class_weight=class_weight,
            verbose=0,
        )
        elapsed    = time.time() - t0
        final_loss = float(hist.history["loss"][-1])
        final_acc  = float(hist.history["accuracy"][-1])

        self._log(
            f"Train local  loss={final_loss:.4f}  acc={final_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )
        return {
            "loss":         final_loss,
            "accuracy":     final_acc,
            "n_samples":    self.n_samples,
            "train_time_s": round(elapsed, 2),
        }

    def evaluate_local(self) -> Dict[str, Any]:
        """Evaluate the current model on the local validation set.

        Returns:
            Metrics dict from evaluate_mlp (accuracy, f1_macro, auc, …).
        """
        metrics = evaluate_mlp(self.model, self.X_val, self.y_val, n_classes=2)
        self._log(
            f"Val local    acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1_macro']:.4f}  "
            f"auc={metrics['auc']:.4f}"
        )
        return metrics

    # ── Kafka ─────────────────────────────────────────────────────────────────

    def _init_kafka(self) -> None:
        """Create the Kafka producer and consumer (called once at start of run)."""
        from kafka import KafkaProducer, KafkaConsumer  # type: ignore

        self._producer = KafkaProducer(
            bootstrap_servers=[self.kafka_broker],
            max_request_size=10 * 1024 * 1024,  # 10 MB headroom
            acks="all",
            retries=3,
        )

        # Unique group_id per client so every client receives the broadcast
        self._consumer = KafkaConsumer(
            TOPIC_GLOBAL,
            bootstrap_servers=[self.kafka_broker],
            group_id=f"fl_client_{self.client_id}_consumer",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            fetch_max_bytes=10 * 1024 * 1024,
        )
        # Force partition assignment so 'latest' offset is locked in before
        # any round starts — prevents missing the first global-model message.
        self._consumer.poll(timeout_ms=200)

        self._log(f"Kafka connecté — broker={self.kafka_broker}")

    def send_weights(self, round_idx: int) -> None:
        """Serialize and publish current weights to the model-weights topic.

        Args:
            round_idx: Current FL round number (embedded in Kafka headers).
        """
        weights = get_weights(self.model)
        payload = serialize_weights(
            weights,
            metadata={
                "client_id": self.client_id,
                "round":     round_idx,
                "n_samples": self.n_samples,
            },
            compress=True,
        )
        headers = [
            (b"round_idx", str(round_idx).encode()),
            (b"client_id", str(self.client_id).encode()),
            (b"n_samples", str(self.n_samples).encode()),
        ]

        future = self._producer.send(
            TOPIC_WEIGHTS,
            value=payload,
            headers=headers,
            key=f"client_{self.client_id}".encode(),
        )
        self._producer.flush()
        record = future.get(timeout=10)

        self._log(
            f"Poids envoyés  round={round_idx}  "
            f"size={len(payload)/1024:.1f} KB  "
            f"offset={record.offset}"
        )

    def receive_global_model(self, round_idx: int, timeout: int = 60) -> bool:
        """Poll for the aggregated global model for the given round.

        The consumer polls in 1-second bursts until either:
          - A message with a matching ``round_idx`` header is found, or
          - The timeout expires, in which case the current weights are kept.

        Args:
            round_idx: Expected round index (matched against Kafka message headers).
            timeout:   Maximum seconds to wait before giving up.

        Returns:
            True if global model received and applied; False on timeout.
        """
        self._log(
            f"En attente du modèle global  round={round_idx}  timeout={timeout}s …"
        )
        deadline = time.time() + timeout

        while time.time() < deadline:
            records = self._consumer.poll(timeout_ms=1_000)
            for _tp, messages in records.items():
                for msg in messages:
                    headers = {k: v.decode() for k, v in (msg.headers or [])}
                    if int(headers.get("round_idx", -1)) == round_idx:
                        weights, _ = deserialize_weights(msg.value)
                        set_weights(self.model, weights)
                        self._log(
                            f"Modèle global reçu  round={round_idx}  "
                            f"fingerprint={get_model_fingerprint(weights)}  "
                            f"size={len(msg.value)/1024:.1f} KB"
                        )
                        return True

        self._log(
            f"Timeout ({timeout}s)  round={round_idx} — "
            "conservation des poids actuels",
            "warning",
        )
        return False

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, repeat_hours: float = 0.0) -> None:
        """Execute n_rounds of federated learning.

        Args:
            repeat_hours: If > 0, restart training after this many hours
                          (must match the server's --repeat-hours value).
        """
        self._log(
            f"Démarrage FL — rounds={self.n_rounds}  "
            f"epochs/round={self.local_epochs}  broker={self.kafka_broker}"
            + (f"  repeat={repeat_hours}h" if repeat_hours > 0 else "")
        )

        self.load_local_data()
        self._build_model()
        self._init_kafka()

        session = 0
        while True:
            session += 1
            round_offset = len(self.history)

            for round_idx in range(1, self.n_rounds + 1):
                cumulative_idx = round_offset + round_idx
                self._log(f"{'─'*10} Round {cumulative_idx} (session {session}, local {round_idx}/{self.n_rounds}) {'─'*10}")
                t_round = time.time()

                train_metrics = self.train_local()
                val_before    = self.evaluate_local()

                self.send_weights(round_idx)
                received  = self.receive_global_model(round_idx)
                val_after = self.evaluate_local() if received else val_before

                self.history.append({
                    "round":            cumulative_idx,
                    "session":          session,
                    "train":            train_metrics,
                    "val_before":       val_before,
                    "val_after":        val_after,
                    "global_received":  received,
                    "round_time_s":     round(time.time() - t_round, 2),
                })

                self._log(
                    f"Round {cumulative_idx} terminé  "
                    f"f1={val_after['f1_macro']:.4f}  "
                    f"auc={val_after['auc']:.4f}  "
                    f"({self.history[-1]['round_time_s']:.1f}s)"
                )

            self._finalize()

            if repeat_hours <= 0:
                break

            self._log(
                f"Session {session} terminée — prochain entraînement dans "
                f"{repeat_hours:.1f}h"
            )
            time.sleep(int(repeat_hours * 3600))

    # ── Finalization ──────────────────────────────────────────────────────────

    def _finalize(self) -> None:
        """Save the final model and per-round history, then close Kafka handles."""
        # ── Final model ───────────────────────────────────────────────────────
        model_path = _MODELS / f"fl_client_{self.client_id}.keras"
        self.model.save(str(model_path))
        self._log(f"Modèle final → {model_path.name}")

        # ── History JSON ──────────────────────────────────────────────────────
        history_path = _ROOT / "data" / f"fl_client_{self.client_id}_history.json"
        payload = {
            "client_id":    self.client_id,
            "n_rounds":     self.n_rounds,
            "local_epochs": self.local_epochs,
            "n_samples":    self.n_samples,
            "history":      self.history,
        }
        with open(history_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        self._log(f"Historique → {history_path.name}")

        # ── Summary ───────────────────────────────────────────────────────────
        final = self.history[-1]["val_after"] if self.history else {}
        self._log(
            f"Résultat final  "
            f"f1_macro={final.get('f1_macro', 0):.4f}  "
            f"auc={final.get('auc', 0):.4f}  "
            f"accuracy={final.get('accuracy', 0):.4f}"
        )

        # ── Close Kafka ───────────────────────────────────────────────────────
        if self._producer:
            self._producer.close()
        if self._consumer:
            self._consumer.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NIDS Federated Learning client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--id",           type=int,   default=1,                  metavar="N",         help="Client ID (1, 2, or 3)")
    p.add_argument("--rounds",       type=int,   default=10,                 metavar="N",         help="FL rounds per session")
    p.add_argument("--epochs",       type=int,   default=5,                  metavar="N",         help="Local training epochs per round")
    p.add_argument("--broker",       type=str,   default="localhost:29092",  metavar="HOST:PORT", help="Kafka bootstrap server")
    p.add_argument("--repeat-hours", type=float, default=0.0,                metavar="H",         help="Repeat training every H hours (0 = run once). Must match server value.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # ── Colored console logging ───────────────────────────────────────────────
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ColorFormatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    # Quiet noisy libraries
    for _lib in ("kafka", "tensorflow", "keras", "absl"):
        logging.getLogger(_lib).setLevel(logging.WARNING)

    client = FLClient(
        client_id=args.id,
        kafka_broker=args.broker,
        n_rounds=args.rounds,
        local_epochs=args.epochs,
    )
    client.run(repeat_hours=args.repeat_hours)
