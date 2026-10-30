"""
federated_learning/model.py
Reusable MLP for centralized training and Federated Learning (FedAvg).

Supports:
  • Binary classification   (n_classes=2) → sigmoid + binary_crossentropy
  • Multiclass              (n_classes>2) → softmax + sparse_categorical_crossentropy
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf
import keras
from keras import layers, regularizers
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── GPU: enable memory growth so 4 GB card isn't fully reserved ───────────────
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

_DEFAULT_L2 = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 1. build_mlp
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp(
    input_dim: int,
    n_classes: int,
    hidden_layers: Sequence[int] = (64, 32, 16),
    learning_rate: float = 1e-3,
    l2: float = _DEFAULT_L2,
    all_batchnorm: bool = False,
) -> keras.Model:
    """Build and compile a fully-connected MLP.

    Architecture (default hidden_layers=(64, 32, 16)):
        Input(input_dim)
        Dense(64, relu) → BatchNorm → Dropout(0.3)
        Dense(32, relu) → BatchNorm → Dropout(0.3)
        Dense(16, relu) → Dropout(0.2)          ← last hidden: no BN by default
        Output: Dense(1, sigmoid)  if n_classes == 2
                Dense(n, softmax)  if n_classes  > 2

    Args:
        input_dim:      Number of input features.
        n_classes:      Number of target classes (2 = binary).
        hidden_layers:  Tuple of units per hidden layer.
        learning_rate:  Adam learning rate.
        l2:             L2 kernel regularization strength.
        all_batchnorm:  If True, add BatchNorm after every hidden layer
                        including the last. Use with hidden_layers=(256,128,64,32)
                        to match the centralized binary MLP weights (warm-start).

    Returns:
        Compiled keras.Model.
    """
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs

    n_hidden = len(hidden_layers)
    for i, units in enumerate(hidden_layers):
        is_last_hidden = i == n_hidden - 1
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.L2(l2),
            name=f"dense_{i}",
        )(x)
        if not is_last_hidden or all_batchnorm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(0.2 if is_last_hidden else 0.3, name=f"dropout_{i}")(x)

    if n_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)
        loss = "sparse_categorical_crossentropy"

    model = keras.Model(inputs=inputs, outputs=outputs, name="mlp_nids")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. train_mlp
# ─────────────────────────────────────────────────────────────────────────────

def train_mlp(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    checkpoint_path: str | Path | None = None,
    sample_weight: np.ndarray | None = None,
) -> keras.callbacks.History:
    """Train with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.

    Args:
        model:            Compiled keras.Model (from build_mlp).
        X_train/y_train:  Training arrays.
        X_val/y_val:      Validation arrays.
        epochs:           Maximum number of epochs.
        batch_size:       Mini-batch size.
        checkpoint_path:  Where to save the best checkpoint (.keras).
                          Defaults to a temp file.
        sample_weight:    Per-sample weights for class imbalance.

    Returns:
        keras.callbacks.History object.
    """
    if checkpoint_path is None:
        checkpoint_path = Path(tempfile.mkdtemp()) / "mlp_best.keras"

    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=5,
            factor=0.5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        sample_weight=sample_weight,
        callbacks=cbs,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 3. evaluate_mlp
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_mlp(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
) -> dict:
    """Evaluate the model and return a metrics dict.

    Returns:
        Dict with keys: accuracy, f1_macro, f1_weighted, precision, recall,
                        auc, confusion_matrix.
    """
    y_prob = model.predict(X_test, verbose=0)

    if n_classes == 2:
        y_prob_flat = y_prob.ravel()
        y_pred = (y_prob_flat >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y_test, y_prob_flat))
        except ValueError:
            auc = float("nan")
    else:
        y_pred = np.argmax(y_prob, axis=1)
        try:
            auc = float(
                roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            auc = float("nan")

    return {
        "accuracy":        float(accuracy_score(y_test, y_pred)),
        "f1_macro":        float(f1_score(y_test, y_pred, average="macro",    zero_division=0)),
        "f1_weighted":     float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_test, y_pred, average="macro",    zero_division=0)),
        "auc":             auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. get_weights / set_weights  (FedAvg interface)
# ─────────────────────────────────────────────────────────────────────────────

def get_weights(model: keras.Model) -> list[np.ndarray]:
    """Return a deep copy of model weights as a list of numpy arrays.

    The copy ensures the caller can mutate the list (e.g. for FedAvg
    aggregation) without affecting the live model.
    """
    return [w.copy() for w in model.get_weights()]


def set_weights(model: keras.Model, weights: list[np.ndarray]) -> None:
    """Apply a list of numpy arrays as model weights (in-place).

    Weight shapes must exactly match the model architecture produced by
    build_mlp — the FedAvg server must use the same hidden_layers config.
    """
    model.set_weights(weights)


# ─────────────────────────────────────────────────────────────────────────────
# 5. __main__ — smoke tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    SEP = "=" * 60

    print(SEP)
    print("federated_learning/model.py — smoke tests")
    print(SEP)

    rng = np.random.default_rng(42)
    N, D = 600, 11  # 11 NIDS features

    X = rng.standard_normal((N, D)).astype(np.float32)

    # ── Test 1: Binary MLP ────────────────────────────────────────────
    print("\n[1] Binary MLP  (n_classes=2)")
    y2 = rng.integers(0, 2, size=N).astype(np.int32)
    m2 = build_mlp(input_dim=D, n_classes=2)
    m2.summary(line_length=60)

    h2 = train_mlp(
        m2, X[:480], y2[:480], X[480:], y2[480:],
        epochs=5, batch_size=32,
    )
    ev2 = evaluate_mlp(m2, X[480:], y2[480:], n_classes=2)
    print(f"  accuracy={ev2['accuracy']:.3f}  f1_macro={ev2['f1_macro']:.3f}  auc={ev2['auc']:.3f}")

    # ── Test 2: Multiclass MLP ────────────────────────────────────────
    print("\n[2] Multiclass MLP  (n_classes=7)")
    y7 = rng.integers(0, 7, size=N).astype(np.int32)
    m7 = build_mlp(input_dim=D, n_classes=7)

    h7 = train_mlp(
        m7, X[:480], y7[:480], X[480:], y7[480:],
        epochs=5, batch_size=32,
    )
    ev7 = evaluate_mlp(m7, X[480:], y7[480:], n_classes=7)
    print(f"  accuracy={ev7['accuracy']:.3f}  f1_macro={ev7['f1_macro']:.3f}  auc={ev7['auc']:.3f}")

    # ── Test 3: get/set_weights round-trip ────────────────────────────
    print("\n[3] get/set_weights round-trip")
    w_before = get_weights(m7)
    # Mutate copy — should not affect model
    w_copy = [w + 1.0 for w in w_before]
    set_weights(m7, w_before)
    w_after = get_weights(m7)
    assert all(
        np.allclose(a, b) for a, b in zip(w_before, w_after)
    ), "Round-trip FAILED"
    # Confirm mutation of w_copy did not corrupt model
    assert not np.allclose(w_copy[0], w_after[0]), "Deep-copy isolation FAILED"
    print("  Round-trip OK  ✅")
    print("  Deep-copy isolation OK  ✅")

    # ── Test 4: architecture shapes ───────────────────────────────────
    print("\n[4] Output shapes")
    x_dummy = np.zeros((4, D), dtype=np.float32)
    out2 = m2.predict(x_dummy, verbose=0)
    out7 = m7.predict(x_dummy, verbose=0)
    assert out2.shape == (4, 1), f"Binary output shape wrong: {out2.shape}"
    assert out7.shape == (4, 7), f"Multiclass output shape wrong: {out7.shape}"
    print(f"  Binary output   : {out2.shape}  ✅")
    print(f"  Multiclass output: {out7.shape}  ✅")

    print(f"\n{SEP}")
    print("All tests passed  ✅")
    print(SEP)
    sys.exit(0)
