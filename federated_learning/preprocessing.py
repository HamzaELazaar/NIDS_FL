"""
federated_learning/preprocessing.py
────────────────────────────────────
Unified preprocessing pipeline for RandomForest, XGBoost, MLP centralisé and FL clients.

Pipeline:
    load_dataset → select_features → create_labels → split_data
    → FeatureScaler.fit_transform → split_for_federated
    → artefacts saved in models/ and data/processed/
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Load .env from project root ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── Coloured logging ──────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_DIM    = "\033[2m"


class _ColouredFormatter(logging.Formatter):
    _LEVEL_COLOURS = {
        logging.DEBUG:    _DIM,
        logging.INFO:     _CYAN,
        logging.WARNING:  _YELLOW,
        logging.ERROR:    _RED,
        logging.CRITICAL: _RED + _BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        colour = self._LEVEL_COLOURS.get(record.levelno, "")
        record.msg = f"{colour}{record.msg}{_RESET}"
        return super().format(record)


def _make_logger(name: str = "preprocessing") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            _ColouredFormatter(
                fmt="%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


log = _make_logger()

# ── Constants ─────────────────────────────────────────────────────────────────
_NORMAL_CLASS   = "Normal Traffic"
_CHUNK_SIZE     = int(os.getenv("PREPROC_CHUNK_SIZE", "300000"))
_N_FL_CLIENTS   = int(os.getenv("FL_NUM_CLIENTS", "3"))
_MODELS_DIR     = _PROJECT_ROOT / "models"
_PROCESSED_DIR  = _PROJECT_ROOT / "data" / "processed"
_CONFIG_PATH    = _PROJECT_ROOT / "data" / "selected_features.json"


# ─────────────────────────────────────────────────────────────────────────────
# 1. load_dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_dir: str = "data/") -> pd.DataFrame:
    """
    Detect all CSV files in *data_dir*, load them in chunks, and return a
    concatenated DataFrame with:
    - stripped column names
    - NaN rows dropped
    - ±inf replaced with 0
    """
    data_path = Path(data_dir) if Path(data_dir).is_absolute() else _PROJECT_ROOT / data_dir
    csv_files = sorted(data_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    log.info(f"📂  {len(csv_files)} CSV file(s) detected in {data_path}")
    for f in csv_files:
        log.debug(f"    {f.name}  ({f.stat().st_size / 1_048_576:.1f} MB)")

    frames: list[pd.DataFrame] = []
    total_rows = 0

    for csv_path in csv_files:
        file_rows = 0
        for chunk in pd.read_csv(csv_path, chunksize=_CHUNK_SIZE, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            chunk.replace([np.inf, -np.inf], 0, inplace=True)
            chunk.dropna(inplace=True)
            frames.append(chunk)
            file_rows += len(chunk)
            total_rows += len(chunk)
        log.info(f"✅  {csv_path.name}  →  {file_rows:,} rows")

    df = pd.concat(frames, ignore_index=True)

    log.info(
        f"{_GREEN}Dataset loaded:{_RESET}  {total_rows:,} rows × {len(df.columns)} cols  "
        f"| memory {df.memory_usage(deep=True).sum() / 1_048_576:.1f} MB"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. select_features
# ─────────────────────────────────────────────────────────────────────────────

def select_features(
    df: pd.DataFrame,
    config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load selected_features.json, keep only the 11 validated features + label
    column, and return (df_reduced, label_col).

    Raises ValueError if any required feature is absent from the DataFrame.
    """
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = json.load(f)

    features: list[str] = config["features"]
    label_col: str = config["label_column"]

    # Verify presence
    missing = [c for c in features + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    df_out = df[features + [label_col]].copy()

    log.info(
        f"✅  Features selected: {len(features)} features + '{label_col}'  "
        f"| {len(df_out):,} rows"
    )
    log.debug(f"    Features: {features}")
    return df_out, label_col


# ─────────────────────────────────────────────────────────────────────────────
# 3. create_labels
# ─────────────────────────────────────────────────────────────────────────────

def create_labels(
    df: pd.DataFrame,
    label_col: str,
    models_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode the label column into:
    - y_multi  : integer class indices (LabelEncoder, 7 classes)
    - y_binary : 0 = Normal Traffic, 1 = Attack (all attack types merged)

    Saves the fitted LabelEncoder to models/label_encoder.joblib.
    Returns (y_multi, y_binary).
    """
    out_dir = Path(models_dir) if models_dir else _MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = df[label_col].values

    le = LabelEncoder()
    y_multi: np.ndarray = le.fit_transform(labels)
    y_binary: np.ndarray = np.where(labels == _NORMAL_CLASS, 0, 1).astype(np.int8)

    # Persist
    le_path = out_dir / "label_encoder.joblib"
    joblib.dump(le, le_path)
    log.info(f"✅  LabelEncoder saved  →  {le_path}")

    # Report
    log.info(f"    Classes ({len(le.classes_)}): {list(le.classes_)}")
    log.info(
        f"    Binary split → Normal: {(y_binary == 0).sum():,}  "
        f"| Attack: {(y_binary == 1).sum():,}"
    )
    for cls, idx in sorted(zip(le.classes_, le.transform(le.classes_))):
        count = (y_multi == idx).sum()
        pct = count / len(y_multi) * 100
        log.debug(f"    [{idx}] {cls:<25}  {count:>9,}  ({pct:.2f}%)")

    return y_multi, y_binary


# ─────────────────────────────────────────────────────────────────────────────
# 4. split_data
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    X: np.ndarray,
    y_multi: np.ndarray,
    y_binary: np.ndarray,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Stratified 70/10/20 split (train / val / test) on y_multi.

    Returns:
        {
          "train": {"X": ..., "y_multi": ..., "y_binary": ...},
          "val":   {"X": ..., "y_multi": ..., "y_binary": ...},
          "test":  {"X": ..., "y_multi": ..., "y_binary": ...},
        }
    """
    # First split: (train+val) vs test
    val_relative = val_size / (1.0 - test_size)   # val fraction within train+val

    X_tv, X_test, ym_tv, ym_test, yb_tv, yb_test = train_test_split(
        X, y_multi, y_binary,
        test_size=test_size, stratify=y_multi, random_state=random_state,
    )

    # Second split: train vs val
    X_train, X_val, ym_train, ym_val, yb_train, yb_val = train_test_split(
        X_tv, ym_tv, yb_tv,
        test_size=val_relative, stratify=ym_tv, random_state=random_state,
    )

    splits = {
        "train": {"X": X_train, "y_multi": ym_train, "y_binary": yb_train},
        "val":   {"X": X_val,   "y_multi": ym_val,   "y_binary": yb_val},
        "test":  {"X": X_test,  "y_multi": ym_test,  "y_binary": yb_test},
    }

    total = len(X)
    for name, split in splits.items():
        n = len(split["X"])
        pct = n / total * 100
        n_attack = split["y_binary"].sum()
        log.info(
            f"✅  {name:<6}  {n:>9,} rows  ({pct:5.1f}%)  "
            f"| attack rate {n_attack/n*100:.2f}%"
        )

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# 5. FeatureScaler
# ─────────────────────────────────────────────────────────────────────────────

class FeatureScaler:
    """
    Thin wrapper around StandardScaler with optional pre-clipping.

    The scaler is fitted **on the training split only** to avoid data leakage.
    Clip bounds are loaded from selected_features.json when available.

    Artefact: models/scaler.joblib
    """

    def __init__(
        self,
        features: list[str] | None = None,
        use_clip: bool = True,
        config_path: str | Path | None = None,
    ) -> None:
        self._scaler  = StandardScaler()
        self._fitted  = False
        self._features = features or []
        self._use_clip = use_clip
        self._clip_bounds: dict[str, tuple[float, float]] = {}

        if use_clip:
            path = Path(config_path) if config_path else _CONFIG_PATH
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    cfg = json.load(f)
                raw = cfg.get("preprocessing", {}).get("clip_bounds", {})
                self._clip_bounds = {
                    feat: (v["lo"], v["hi"]) for feat, v in raw.items()
                }
                log.debug(f"    Clip bounds loaded for {len(self._clip_bounds)} features")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _clip(self, X: np.ndarray) -> np.ndarray:
        """Apply per-feature clipping using bounds from config."""
        if not self._clip_bounds or not self._features:
            return X
        X_out = X.copy().astype(np.float64)
        for i, feat in enumerate(self._features):
            if feat in self._clip_bounds:
                lo, hi = self._clip_bounds[feat]
                X_out[:, i] = np.clip(X_out[:, i], lo, hi)
        return X_out

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Clip + fit StandardScaler on X_train, then return scaled array."""
        X_clipped = self._clip(X_train)
        X_scaled  = self._scaler.fit_transform(X_clipped)
        self._fitted = True
        log.info(
            f"✅  FeatureScaler fitted  |  mean range "
            f"[{self._scaler.mean_.min():.3f}, {self._scaler.mean_.max():.3f}]"
        )
        return X_scaled.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip + scale with the already-fitted scaler."""
        if not self._fitted:
            raise RuntimeError("FeatureScaler must be fitted before calling transform().")
        X_clipped = self._clip(X)
        return self._scaler.transform(X_clipped).astype(np.float32)

    def save(self, models_dir: str | Path | None = None) -> Path:
        """Persist to models/scaler.joblib. Returns the saved path."""
        if not self._fitted:
            raise RuntimeError("FeatureScaler is not fitted — nothing to save.")
        out_dir = Path(models_dir) if models_dir else _MODELS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "scaler.joblib"
        joblib.dump(self, path)
        log.info(f"✅  FeatureScaler saved  →  {path}")
        return path

    @classmethod
    def load(cls, models_dir: str | Path | None = None) -> "FeatureScaler":
        """Load a previously saved FeatureScaler from models/scaler.joblib."""
        path = (Path(models_dir) if models_dir else _MODELS_DIR) / "scaler.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Scaler not found: {path}")
        scaler: FeatureScaler = joblib.load(path)
        log.info(f"✅  FeatureScaler loaded  ←  {path}")
        return scaler


# ─────────────────────────────────────────────────────────────────────────────
# 6. split_for_federated
# ─────────────────────────────────────────────────────────────────────────────

def split_for_federated(
    X: np.ndarray,
    y_binary: np.ndarray,
    n_clients: int | None = None,
    mode: Literal["iid", "non_iid"] = "iid",
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Partition (X, y_binary) across *n_clients* FL clients.

    iid     — random shuffle, equal-size partitions (same class distribution).
    non_iid — Dirichlet(α=0.5) sampling: each client receives a biased mix of
              attack types, simulating heterogeneous real-world sensors.

    Returns a list of (X_client, y_client) tuples.
    """
    n = n_clients or _N_FL_CLIENTS
    rng = np.random.default_rng(random_state)

    if mode == "iid":
        idx = rng.permutation(len(X))
        partitions = np.array_split(idx, n)
        clients = [(X[p], y_binary[p]) for p in partitions]

        log.info(f"✅  IID split  →  {n} clients")
        for i, (xc, yc) in enumerate(clients):
            log.debug(
                f"    client {i}  {len(xc):>9,} rows  "
                f"| attack rate {yc.mean()*100:.2f}%"
            )

    elif mode == "non_iid":
        # Separate indices by class (0=normal, 1=attack)
        normal_idx  = np.where(y_binary == 0)[0]
        attack_idx  = np.where(y_binary == 1)[0]

        rng.shuffle(normal_idx)
        rng.shuffle(attack_idx)

        # Split normal traffic evenly (every client needs baseline knowledge)
        normal_parts = np.array_split(normal_idx, n)

        # Dirichlet distribution for attack samples across clients
        alpha = np.full(n, 0.5)
        proportions = rng.dirichlet(alpha)
        proportions = (proportions * len(attack_idx)).astype(int)
        # Correct rounding error on the last client
        proportions[-1] = len(attack_idx) - proportions[:-1].sum()

        attack_splits: list[np.ndarray] = []
        start = 0
        for size in proportions:
            attack_splits.append(attack_idx[start : start + size])
            start += size

        clients = []
        for i in range(n):
            part_idx = np.concatenate([normal_parts[i], attack_splits[i]])
            rng.shuffle(part_idx)
            clients.append((X[part_idx], y_binary[part_idx]))

        log.info(f"✅  Non-IID split (Dirichlet α=0.5)  →  {n} clients")
        for i, (xc, yc) in enumerate(clients):
            log.debug(
                f"    client {i}  {len(xc):>9,} rows  "
                f"| attack rate {yc.mean()*100:.2f}%"
            )

    else:
        raise ValueError(f"mode must be 'iid' or 'non_iid', got '{mode}'")

    return clients


# ─────────────────────────────────────────────────────────────────────────────
# 7. Helpers: save / load processed splits
# ─────────────────────────────────────────────────────────────────────────────

def save_splits(
    splits: dict[str, dict[str, np.ndarray]],
    processed_dir: str | Path | None = None,
) -> None:
    """Save train/val/test splits to data/processed/{split}.npz."""
    out_dir = Path(processed_dir) if processed_dir else _PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arrays in splits.items():
        path = out_dir / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        size_kb = path.stat().st_size / 1024
        log.info(f"✅  {name}.npz saved  →  {path}  ({size_kb:.0f} KB)")


def load_splits(
    processed_dir: str | Path | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Load pre-saved splits from data/processed/. Raises if any file is missing."""
    in_dir = Path(processed_dir) if processed_dir else _PROCESSED_DIR
    splits: dict[str, dict[str, np.ndarray]] = {}
    for name in ("train", "val", "test"):
        path = in_dir / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Split not found: {path} — run preprocessing first")
        data = np.load(path)
        splits[name] = {k: data[k] for k in data.files}
        log.debug(f"    {name}.npz loaded  |  X shape {splits[name]['X'].shape}")
    log.info("✅  All splits loaded from data/processed/")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# 8. __main__ — full pipeline
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    _SEP = "═" * 62

    def _section(title: str) -> None:
        print(f"\n{_BOLD}{_CYAN}{_SEP}{_RESET}")
        print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
        print(f"{_BOLD}{_CYAN}{_SEP}{_RESET}")

    t_start = time.time()

    print(f"\n{_BOLD}{_GREEN}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     NIDS-FL — Preprocessing Pipeline                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{_RESET}")

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    _section("1 / 6  —  Load dataset")
    df = load_dataset("data/")
    print(f"  Shape         : {df.shape}")
    print(f"  Memory        : {df.memory_usage(deep=True).sum() / 1_048_576:.1f} MB")

    # ── Step 2: Select features ───────────────────────────────────────────────
    _section("2 / 6  —  Select 11 features")
    df_sel, label_col = select_features(df)
    del df   # free memory
    print(f"  Retained shape: {df_sel.shape}")
    print(f"  Label column  : '{label_col}'")

    X_raw = df_sel.drop(columns=[label_col]).values.astype(np.float64)

    # ── Step 3: Labels ────────────────────────────────────────────────────────
    _section("3 / 6  —  Create labels")
    y_multi, y_binary = create_labels(df_sel, label_col)
    print(f"  y_multi  shape: {y_multi.shape}   dtype: {y_multi.dtype}")
    print(f"  y_binary shape: {y_binary.shape}  dtype: {y_binary.dtype}")
    print(f"  Normal  : {(y_binary == 0).sum():>9,}  ({(y_binary==0).mean()*100:.2f}%)")
    print(f"  Attack  : {(y_binary == 1).sum():>9,}  ({(y_binary==1).mean()*100:.2f}%)")

    # ── Step 4: Split ─────────────────────────────────────────────────────────
    _section("4 / 6  —  Stratified split (70 / 10 / 20)")
    splits_raw = split_data(X_raw, y_multi, y_binary)
    for name, s in splits_raw.items():
        print(f"  {name:<6}  X={s['X'].shape}  "
              f"y_multi={s['y_multi'].shape}  "
              f"y_binary={s['y_binary'].shape}")

    # ── Step 5: Scale ─────────────────────────────────────────────────────────
    _section("5 / 6  —  Clip + StandardScale")

    with open(_CONFIG_PATH, encoding="utf-8") as f:
        _cfg = json.load(f)
    feature_names: list[str] = _cfg["features"]

    scaler = FeatureScaler(features=feature_names, use_clip=True)
    splits_scaled: dict[str, dict[str, np.ndarray]] = {}

    splits_scaled["train"] = {
        "X":        scaler.fit_transform(splits_raw["train"]["X"]),
        "y_multi":  splits_raw["train"]["y_multi"],
        "y_binary": splits_raw["train"]["y_binary"],
    }
    for name in ("val", "test"):
        splits_scaled[name] = {
            "X":        scaler.transform(splits_raw[name]["X"]),
            "y_multi":  splits_raw[name]["y_multi"],
            "y_binary": splits_raw[name]["y_binary"],
        }
    scaler.save()

    print(f"  Train X  : mean={splits_scaled['train']['X'].mean():.4f}"
          f"  std={splits_scaled['train']['X'].std():.4f}")
    print(f"  Val   X  : mean={splits_scaled['val']['X'].mean():.4f}"
          f"  std={splits_scaled['val']['X'].std():.4f}")
    print(f"  Test  X  : mean={splits_scaled['test']['X'].mean():.4f}"
          f"  std={splits_scaled['test']['X'].std():.4f}")

    # ── Step 6: Save splits ───────────────────────────────────────────────────
    _section("6 / 6  —  Save artefacts")
    save_splits(splits_scaled)

    # Quick federated preview (IID + non-IID)
    X_tr = splits_scaled["train"]["X"]
    y_tr = splits_scaled["train"]["y_binary"]

    print(f"\n  {_CYAN}Federated IID ({_N_FL_CLIENTS} clients):{_RESET}")
    iid_clients = split_for_federated(X_tr, y_tr, mode="iid")
    for i, (xc, yc) in enumerate(iid_clients):
        print(f"    client {i}  {xc.shape}  attack={yc.mean()*100:.2f}%")

    print(f"\n  {_CYAN}Federated Non-IID ({_N_FL_CLIENTS} clients):{_RESET}")
    niid_clients = split_for_federated(X_tr, y_tr, mode="non_iid")
    for i, (xc, yc) in enumerate(niid_clients):
        print(f"    client {i}  {xc.shape}  attack={yc.mean()*100:.2f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{_BOLD}{_GREEN}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   ✅  Pipeline complete                                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{_RESET}")
    print(f"  Total time     : {elapsed:.1f}s")
    print(f"  Artefacts saved:")
    print(f"    models/label_encoder.joblib")
    print(f"    models/scaler.joblib")
    print(f"    data/processed/train.npz   ({splits_scaled['train']['X'].shape[0]:,} rows)")
    print(f"    data/processed/val.npz     ({splits_scaled['val']['X'].shape[0]:,} rows)")
    print(f"    data/processed/test.npz    ({splits_scaled['test']['X'].shape[0]:,} rows)")
    print()
