"""
federated_learning/aggregation.py
----------------------------------
Implémentation de FedAvg (McMahan et al., 2017) pour agréger
les poids de modèles Keras distribués via Apache Kafka.

Référence :
  McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
  Communication-efficient learning of deep networks from decentralized data.
  AISTATS 2017.
"""

import base64
import gzip
import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FedAvg
# ─────────────────────────────────────────────────────────────────────────────

def federated_average(
    client_weights: List[List[np.ndarray]],
    client_sizes: List[int],
    min_clients: int = 2,
) -> List[np.ndarray]:
    """Federated Averaging (FedAvg) — McMahan et al., 2017.

    Computes the weighted average of client model weights proportional to
    each client's local dataset size:

        w_global = Σ_k (n_k / N) * w_k

    where n_k is the number of local samples for client k and N = Σ n_k.

    Args:
        client_weights: Weights per client — shape [n_clients][n_layers][*layer_shape].
        client_sizes:   Number of local training samples per client.
        min_clients:    Minimum number of clients required before aggregating.

    Returns:
        Aggregated global weights — same structure as a single client's weight list.

    Raises:
        ValueError: Fewer clients than min_clients, mismatched lengths, or
                    incompatible architectures across clients.
    """
    if len(client_weights) < min_clients:
        raise ValueError(
            f"Reçu {len(client_weights)} clients, minimum requis : {min_clients}"
        )

    if len(client_weights) != len(client_sizes):
        raise ValueError(
            f"Mismatch : {len(client_weights)} poids mais "
            f"{len(client_sizes)} tailles de clients"
        )

    n_layers = len(client_weights[0])
    for k, weights in enumerate(client_weights):
        if len(weights) != n_layers:
            raise ValueError(
                f"Client {k} a {len(weights)} layers, attendu {n_layers}"
            )
        for layer_idx, (w_ref, w_k) in enumerate(zip(client_weights[0], weights)):
            if w_ref.shape != w_k.shape:
                raise ValueError(
                    f"Client {k}, layer {layer_idx} : shape {w_k.shape} "
                    f"ne correspond pas à {w_ref.shape}"
                )

    N_total = sum(client_sizes)
    coefficients = [n_k / N_total for n_k in client_sizes]

    logger.info(
        "FedAvg : %d clients, N_total=%d, coefficients=%s",
        len(client_weights),
        N_total,
        [f"{c:.3f}" for c in coefficients],
    )

    # float64 accumulation → float32 storage (avoids cumulative rounding errors)
    global_weights: List[np.ndarray] = []
    for layer_idx in range(n_layers):
        layer_agg = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)
        for weights, coeff in zip(client_weights, coefficients):
            layer_agg += coeff * weights[layer_idx].astype(np.float64)
        global_weights.append(layer_agg.astype(np.float32))

    return global_weights


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sérialisation / Désérialisation
# ─────────────────────────────────────────────────────────────────────────────

def serialize_weights(
    weights: List[np.ndarray],
    metadata: Optional[Dict[str, Any]] = None,
    compress: bool = True,
) -> bytes:
    """Serialize model weights to bytes for Kafka transport.

    Internal format (pickle-serialized dict, optionally gzip-compressed):

        {
            "metadata"       : { client_id, n_samples, round, … },
            "weights_shapes" : [[11, 64], [64], …],
            "weights_dtypes" : ["float32", …],
            "weights"        : [<base64-encoded float32 bytes>, …],
            "n_layers"       : int,
            "total_params"   : int,
            "checksum"       : "<md5-hex>",
            "serialized_at"  : float (unix timestamp),
        }

    Args:
        weights:  Keras model weights as list of numpy arrays.
        metadata: Optional dict with client/round info to embed in the payload.
        compress: Apply gzip compression (level 6). Reduces size ~60–70%.

    Returns:
        Serialized bytes ready to publish to a Kafka topic.
    """
    payload: Dict[str, Any] = {
        "metadata":       metadata or {},
        "weights_shapes": [list(w.shape) for w in weights],
        "weights_dtypes": [str(w.dtype) for w in weights],
        "weights": [
            base64.b64encode(w.astype(np.float32).tobytes()).decode()
            for w in weights
        ],
        "n_layers":     len(weights),
        "total_params": sum(w.size for w in weights),
        "serialized_at": time.time(),
    }

    flat = np.concatenate([w.flatten().astype(np.float32) for w in weights])
    payload["checksum"] = hashlib.md5(flat.tobytes()).hexdigest()

    raw = pickle.dumps(payload)

    if compress:
        result = gzip.compress(raw, compresslevel=6)
        logger.debug(
            "Sérialisé : %.1f KB → %.1f KB (gzip, %d%% réduction)",
            len(raw) / 1024,
            len(result) / 1024,
            100 * (1 - len(result) / len(raw)),
        )
    else:
        result = raw

    return result


def deserialize_weights(
    data: bytes,
    verify_checksum: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Deserialize bytes received from Kafka back into numpy weight arrays.

    Args:
        data:             Raw bytes from a Kafka message value.
        verify_checksum:  Validate MD5 integrity of reconstructed weights.

    Returns:
        Tuple of (weights, metadata) where weights is a list of numpy arrays
        and metadata is the dict embedded at serialization time.

    Raises:
        ValueError: Checksum mismatch (data corrupted in transit).
        ValueError: Malformed payload (missing required keys).
    """
    try:
        raw = gzip.decompress(data)
    except OSError:
        raw = data  # not compressed

    payload: Dict[str, Any] = pickle.loads(raw)

    required_keys = {"weights_shapes", "weights_dtypes", "weights"}
    missing = required_keys - payload.keys()
    if missing:
        raise ValueError(f"Payload invalide — clés manquantes : {missing}")

    weights: List[np.ndarray] = []
    for shape, w_b64 in zip(payload["weights_shapes"], payload["weights"]):
        w_bytes = base64.b64decode(w_b64)
        w = np.frombuffer(w_bytes, dtype=np.float32).reshape(shape)
        weights.append(w)

    if verify_checksum:
        flat = np.concatenate([w.flatten().astype(np.float32) for w in weights])
        actual = hashlib.md5(flat.tobytes()).hexdigest()
        expected = payload.get("checksum", "")
        if actual != expected:
            raise ValueError(
                f"Checksum invalide : {actual} != {expected}. "
                "Données corrompues en transit."
            )

    metadata: Dict[str, Any] = payload.get("metadata", {})
    logger.debug(
        "Désérialisé : %d layers, %s paramètres, client=%s",
        payload.get("n_layers", len(weights)),
        f"{payload.get('total_params', 0):,}",
        metadata.get("client_id", "?"),
    )

    return weights, metadata


# ─────────────────────────────────────────────────────────────────────────────
# 3. Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def compute_weight_delta(
    weights_before: List[np.ndarray],
    weights_after: List[np.ndarray],
) -> Dict[str, Any]:
    """Compute per-layer and aggregate weight change statistics between rounds.

    A mean_delta approaching 0 indicates convergence.

    Args:
        weights_before: Weights at the start of a round.
        weights_after:  Weights at the end of a round.

    Returns:
        Dict with keys: mean_delta, max_delta, layer_deltas (list), converged (bool).
    """
    layer_deltas = [
        float(np.abs(w_after.astype(np.float64) - w_before.astype(np.float64)).mean())
        for w_before, w_after in zip(weights_before, weights_after)
    ]
    mean_delta = float(np.mean(layer_deltas))
    return {
        "mean_delta":   mean_delta,
        "max_delta":    float(np.max(layer_deltas)),
        "layer_deltas": layer_deltas,
        "converged":    mean_delta < 1e-4,
    }


def get_model_fingerprint(weights: List[np.ndarray]) -> str:
    """Return a 16-character SHA-256 hex digest of flattened model weights.

    Useful for verifying that all clients share the same global model after
    each aggregation round.
    """
    flat = np.concatenate([w.flatten().astype(np.float32) for w in weights])
    return hashlib.sha256(flat.tobytes()).hexdigest()[:16]


def weights_size_kb(weights: List[np.ndarray]) -> float:
    """Return the in-memory size of the weight list in KB (float32 layout)."""
    return sum(w.size for w in weights) * 4 / 1024


def serialize_size_kb(weights: List[np.ndarray], compress: bool = True) -> float:
    """Return the actual serialized size of weights in KB (after gzip if compress)."""
    return len(serialize_weights(weights, compress=compress)) / 1024


# ─────────────────────────────────────────────────────────────────────────────
# 4. Chargement des poids initiaux
# ─────────────────────────────────────────────────────────────────────────────

def load_initial_weights(
    weights_path: str = "models/final/best_mlp_binary_weights_init.pkl",
) -> List[np.ndarray]:
    """Load the pre-trained centralized MLP weights for FL warm-start.

    Using centralized weights as the starting point for all FL clients avoids
    cold-start and significantly accelerates convergence (fewer rounds needed
    to reach the target F1).

    Args:
        weights_path: Path to the pickle file produced by save_best_models.py.

    Returns:
        List of numpy arrays representing the model's initial weights.

    Raises:
        FileNotFoundError: The pickle file does not exist at weights_path.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Poids initiaux introuvables : {weights_path}\n"
            "Lancer d'abord : python scripts/save_best_models.py"
        )

    with open(path, "rb") as f:
        weights: List[np.ndarray] = pickle.load(f)

    logger.info(
        "Poids initiaux chargés depuis %s : %d layers, %s paramètres, %.1f KB",
        weights_path,
        len(weights),
        f"{sum(w.size for w in weights):,}",
        weights_size_kb(weights),
    )

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# 5. __main__ — tests autonomes
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    SEP = "=" * 65

    print(SEP)
    print("  Test federated_learning/aggregation.py")
    print(SEP)

    rng = np.random.default_rng(42)

    # MLP binaire : Dense(11→64) + bias + Dense(64→32) + bias +
    #               Dense(32→16) + bias + Dense(16→1) + bias
    layer_shapes = [
        (11, 64), (64,),
        (64, 32), (32,),
        (32, 16), (16,),
        (16,  1), (1,),
    ]
    client_sizes = [10_000, 8_000, 12_000]

    # ── Test 1 : FedAvg avec 3 clients simulés ──────────────────────────────
    print("\n[Test 1] FedAvg — 3 clients simulés")

    clients_w: List[List[np.ndarray]] = [
        [rng.standard_normal(s).astype(np.float32) * (1 + k * 0.1) for s in layer_shapes]
        for k, _ in enumerate(client_sizes)
    ]

    global_w = federated_average(clients_w, client_sizes)
    print(f"  ✅ FedAvg OK — {len(global_w)} layers agrégés")
    print(f"     Layer 0 shape : {global_w[0].shape}")

    N = sum(client_sizes)
    expected_layer0 = sum(
        (n / N) * w[0] for n, w in zip(client_sizes, clients_w)
    )
    assert np.allclose(global_w[0], expected_layer0, atol=1e-5), "FedAvg incorrect !"
    print("  ✅ Moyenne pondérée vérifiée mathématiquement")

    # ── Test 2 : Sérialisation / Désérialisation ─────────────────────────────
    print("\n[Test 2] Sérialisation / Désérialisation")

    meta_in: Dict[str, Any] = {"client_id": 1, "round": 3, "n_samples": 10_000}
    serialized = serialize_weights(global_w, metadata=meta_in, compress=True)
    raw_kb  = weights_size_kb(global_w)
    ser_kb  = len(serialized) / 1024

    print(f"  Poids bruts  : {raw_kb:.1f} KB")
    print(f"  Sérialisé    : {ser_kb:.1f} KB")
    assert ser_kb < 250, f"Taille sérialisée trop grande : {ser_kb:.1f} KB ≥ 250 KB"

    weights_back, meta_out = deserialize_weights(serialized, verify_checksum=True)
    assert meta_out["client_id"] == 1
    for w_orig, w_back in zip(global_w, weights_back):
        assert np.allclose(w_orig, w_back, atol=1e-6), "Round-trip échoué !"
    print("  ✅ Round-trip sérialisation parfait (checksum OK)")
    print(f"  ✅ Taille sérialisée OK : {ser_kb:.1f} KB < 250 KB")

    # ── Test 3 : Fingerprint et convergence ──────────────────────────────────
    print("\n[Test 3] Fingerprint et delta de convergence")

    fp = get_model_fingerprint(global_w)
    fp_back = get_model_fingerprint(weights_back)
    assert fp == fp_back, "Fingerprints divergent après round-trip !"
    print(f"  Fingerprint  : {fp}")

    delta = compute_weight_delta(global_w, weights_back)
    print(f"  Mean delta   : {delta['mean_delta']:.2e}")
    print(f"  Converged    : {delta['converged']}")
    assert delta["mean_delta"] < 1e-6, "Delta trop grand pour un round-trip identique"
    print("  ✅ Utilitaires OK")

    # ── Test 4 : Agrégation partielle (fault tolerance) ──────────────────────
    print("\n[Test 4] Fault tolerance — 2 clients sur 3")
    try:
        partial_w = federated_average(clients_w[:2], client_sizes[:2], min_clients=2)
        print(f"  ✅ Agrégation partielle OK ({len(clients_w[:2])} clients)")
    except ValueError as exc:
        print(f"  ❌ Erreur inattendue : {exc}")
        sys.exit(1)

    # ── Test 5 : Architecture incompatible → ValueError ──────────────────────
    print("\n[Test 5] Validation architecture incompatible")
    bad_weights = [list(w) for w in clients_w]
    bad_weights[1][0] = rng.standard_normal((11, 512)).astype(np.float32)
    try:
        federated_average(bad_weights, client_sizes)
        print("  ❌ Erreur : ValueError aurait dû être levé")
        sys.exit(1)
    except ValueError as exc:
        print(f"  ✅ ValueError correctement levé : {exc}")

    # ── Test 6 : Chargement poids initiaux ───────────────────────────────────
    print("\n[Test 6] Chargement poids initiaux")
    try:
        init_w = load_initial_weights()
        print(f"  ✅ {len(init_w)} layers chargés")
        print(f"     {sum(w.size for w in init_w):,} paramètres")
        print(f"     Fingerprint : {get_model_fingerprint(init_w)}")
    except FileNotFoundError as exc:
        print("  ⚠️  Fichier manquant (normal si save_best_models.py pas encore lancé)")
        print(f"     {exc}")

    print(f"\n{SEP}")
    print("  ✅ Tous les tests passés — aggregation.py opérationnel")
    print(SEP)
    sys.exit(0)
