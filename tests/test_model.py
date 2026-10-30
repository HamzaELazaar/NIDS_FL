"""tests/test_model.py — Validate that saved Keras models load and predict correctly."""
import numpy as np
import pytest
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
INPUT_DIM  = 11
N_CLASSES  = 7


@pytest.fixture(scope="module")
def mlp_multi():
    keras = pytest.importorskip("keras")
    path = MODELS_DIR / "mlp_multi.keras"
    assert path.exists(), f"mlp_multi.keras not found at {path}"
    return keras.models.load_model(str(path))


@pytest.fixture(scope="module")
def mlp_binary():
    keras = pytest.importorskip("keras")
    path = MODELS_DIR / "mlp_binary.keras"
    assert path.exists(), f"mlp_binary.keras not found at {path}"
    return keras.models.load_model(str(path))


def _dummy(n=16):
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, INPUT_DIM)).astype(np.float32)


# ── mlp_multi tests ───────────────────────────────────────────────────────────

def test_multi_loads(mlp_multi):
    assert mlp_multi is not None


def test_multi_input_shape(mlp_multi):
    assert mlp_multi.input_shape == (None, INPUT_DIM)


def test_multi_output_shape(mlp_multi):
    assert mlp_multi.output_shape == (None, N_CLASSES)


def test_multi_predict_shape(mlp_multi):
    out = mlp_multi.predict(_dummy(), verbose=0)
    assert out.shape == (16, N_CLASSES)


def test_multi_probabilities_sum_to_one(mlp_multi):
    out = mlp_multi.predict(_dummy(), verbose=0)
    row_sums = out.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), f"Row sums: {row_sums}"


def test_multi_predict_finite(mlp_multi):
    out = mlp_multi.predict(_dummy(), verbose=0)
    assert np.all(np.isfinite(out)), "NaN/Inf in multiclass predictions"


# ── mlp_binary tests ──────────────────────────────────────────────────────────

def test_binary_loads(mlp_binary):
    assert mlp_binary is not None


def test_binary_input_shape(mlp_binary):
    assert mlp_binary.input_shape == (None, INPUT_DIM)


def test_binary_output_shape(mlp_binary):
    assert mlp_binary.output_shape == (None, 1)


def test_binary_predict_range(mlp_binary):
    out = mlp_binary.predict(_dummy(), verbose=0)
    assert out.shape == (16, 1)
    assert np.all(out >= 0.0) and np.all(out <= 1.0), "Sigmoid output out of [0,1]"


def test_binary_predict_finite(mlp_binary):
    out = mlp_binary.predict(_dummy(), verbose=0)
    assert np.all(np.isfinite(out)), "NaN/Inf in binary predictions"


# ── mlp_binary_weights_init.pkl ───────────────────────────────────────────────

def test_weights_init_exists():
    path = MODELS_DIR / "mlp_binary_weights_init.pkl"
    assert path.exists(), f"Weights init file not found at {path}"


def test_weights_init_round_trip(mlp_binary):
    import pickle
    path = MODELS_DIR / "mlp_binary_weights_init.pkl"
    with open(path, "rb") as f:
        weights = pickle.load(f)
    assert isinstance(weights, list)
    assert all(isinstance(w, np.ndarray) for w in weights)
    # Confirm shapes match model
    model_weights = mlp_binary.get_weights()
    assert len(weights) == len(model_weights), "Weight list length mismatch"
