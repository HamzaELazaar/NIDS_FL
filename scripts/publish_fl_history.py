#!/usr/bin/env python3
"""
publish_fl_history.py — replay saved FL training data into Kafka without retraining.

Publishes to:
  fl-rounds     — round metrics from data/fl_history.json
  global-model  — best model weights from models/final/best_mlp_binary_weights_init.pkl
  model-weights — same weights (treated as the final client snapshot)

Usage:
    python scripts/publish_fl_history.py
    python scripts/publish_fl_history.py --broker localhost:29092 --delay 0.3
"""

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

BROKER         = "localhost:29092"
TOPIC_ROUNDS   = "fl-rounds"
TOPIC_GLOBAL   = "global-model"
TOPIC_WEIGHTS  = "model-weights"
HISTORY_FILE   = _ROOT / "data"   / "fl_history.json"
WEIGHTS_PKL    = _ROOT / "models" / "final" / "best_mlp_binary_weights_init.pkl"
MANIFEST_FILE  = _ROOT / "models" / "final" / "MANIFEST.json"


def _make_producer(broker: str):
    from kafka import KafkaProducer
    return KafkaProducer(
        bootstrap_servers=[broker],
        acks="all",
        linger_ms=5,
    )


def publish_rounds(producer, history: list, delay: float) -> None:
    print(f"\n[fl-rounds] Publishing {len(history)} rounds …")
    for record in history:
        payload = {
            "round":              record.get("round", 0),
            "global_f1":          record.get("global_f1",       record.get("f1_macro",  0.0)),
            "global_accuracy":    record.get("global_accuracy", record.get("accuracy",  0.0)),
            "global_loss":        record.get("global_loss",     record.get("loss",       0.0)),
            "aggregation_method": record.get("aggregation_method", "FedAvg"),
            "timestamp":          record.get("timestamp", ""),
            "clients":            record.get("clients", []),
        }
        producer.send(
            TOPIC_ROUNDS,
            key=str(payload["round"]).encode(),
            value=json.dumps(payload, default=str).encode("utf-8"),
        )
        print(f"  round={payload['round']:>2}  f1={payload['global_f1']:.4f}  "
              f"acc={payload['global_accuracy']:.4f}  loss={payload['global_loss']:.4f}")
        time.sleep(delay)
    producer.flush()
    print(f"[fl-rounds] Done — {len(history)} rounds published.")


def publish_global_model(producer, weights_pkl: Path, manifest: dict) -> None:
    from federated_learning.aggregation import load_initial_weights, serialize_weights, get_model_fingerprint

    print(f"\n[global-model] Loading weights from {weights_pkl.name} …")
    weights = load_initial_weights(str(weights_pkl))

    n_rounds = manifest.get("best_edge", {}).get("round", 10)
    fingerprint = get_model_fingerprint(weights)
    payload = serialize_weights(
        weights,
        metadata={"source": "server", "round": n_rounds},
        compress=True,
    )

    producer.send(
        TOPIC_GLOBAL,
        key=b"global",
        value=payload,
        headers=[
            ("round_idx", str(n_rounds).encode()),
            ("source",    b"server"),
        ],
    )
    producer.flush()
    print(f"[global-model] Published  size={len(payload)/1024:.1f} KB  "
          f"fingerprint={fingerprint}  round={n_rounds}")


def publish_model_weights(producer, weights_pkl: Path) -> None:
    from federated_learning.aggregation import load_initial_weights, serialize_weights

    print(f"\n[model-weights] Publishing client snapshot from {weights_pkl.name} …")
    weights = load_initial_weights(str(weights_pkl))

    for client_id in [1, 2, 3]:
        payload = serialize_weights(
            weights,
            metadata={"source": "client", "client_id": client_id, "round": 10},
            compress=True,
        )
        producer.send(
            TOPIC_WEIGHTS,
            key=str(client_id).encode(),
            value=payload,
            headers=[
                ("round_idx",  b"10"),
                ("client_id",  str(client_id).encode()),
            ],
        )
        print(f"  client={client_id}  size={len(payload)/1024:.1f} KB")
    producer.flush()
    print("[model-weights] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay FL history into Kafka")
    parser.add_argument("--broker", default=BROKER)
    parser.add_argument("--delay",  type=float, default=0.1,
                        help="Seconds between round publishes (default 0.1)")
    parser.add_argument("--rounds-only", action="store_true",
                        help="Only publish fl-rounds, skip model weight topics")
    args = parser.parse_args()

    if not HISTORY_FILE.exists():
        print(f"ERROR: {HISTORY_FILE} not found"); sys.exit(1)
    if not WEIGHTS_PKL.exists():
        print(f"ERROR: {WEIGHTS_PKL} not found"); sys.exit(1)

    history = json.loads(HISTORY_FILE.read_text())
    if not isinstance(history, list):
        print("ERROR: fl_history.json must be a list of round records"); sys.exit(1)

    manifest = {}
    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text())

    print(f"NIDS-FL Kafka history replay")
    print(f"  Broker  : {args.broker}")
    print(f"  Rounds  : {len(history)}")
    print(f"  Weights : {WEIGHTS_PKL.name}")

    from kafka.errors import NoBrokersAvailable
    try:
        producer = _make_producer(args.broker)
    except NoBrokersAvailable:
        print(f"ERROR: Kafka not reachable at {args.broker}"); sys.exit(1)

    try:
        publish_rounds(producer, history, args.delay)

        if not args.rounds_only:
            publish_global_model(producer, WEIGHTS_PKL, manifest)
            publish_model_weights(producer, WEIGHTS_PKL)

        print("\nAll done — check the dashboard FL page and Kafka UI.")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
