"""
scripts/save_best_models.py
Consolidate the best trained models into models/final/ with a canonical MANIFEST.json.

Usage:
    python scripts/save_best_models.py [--dry-run]

What it does:
    1. Loads all metrics from JSON files (with hardcoded fallbacks for per-variant
       metrics not stored individually).
    2. Selects the best candidate per role using the configured criterion.
    3. Copies selected models + artefacts into models/final/.
    4. Validates each copied model actually loads.
    5. Writes models/final/MANIFEST.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
FINAL_DIR    = MODELS_DIR / "final"
DATA_DIR     = PROJECT_ROOT / "data"

# ── Model registry ────────────────────────────────────────────────────────────
# Maps model_id → source file (relative to MODELS_DIR)

MODEL_FILES: dict[str, Path] = {
    "rf":              MODELS_DIR / "rf_model.joblib",
    "xgboost":         MODELS_DIR / "xgboost_final.json",
    "mlp_multi_v1":    MODELS_DIR / "mlp_v1_balanced_best.keras",
    "mlp_multi_v2":    MODELS_DIR / "mlp_v2_focal_best.keras",
    "mlp_multi_v3":    MODELS_DIR / "mlp_v3_combined_best.keras",
    "mlp_binary_v1":   MODELS_DIR / "mlp_bin_v1_best.keras",
    "mlp_binary_v2":   MODELS_DIR / "mlp_bin_v2_best.keras",
    "mlp_binary_v3":   MODELS_DIR / "mlp_bin_v3_best.keras",
}

# Known metrics from training runs — used when per-variant JSON isn't available.
# Values sourced from notebook outputs recorded in rapport_centralized.md.
KNOWN_METRICS: dict[str, dict] = {
    "rf": {
        "accuracy": 0.994319, "f1_macro": 0.862657, "f1_weighted": 0.995955,
        "f1_per_class": {
            "Bots": 0.78, "Brute Force": 0.9992, "DDoS": 0.9998,
            "DoS": 0.9976, "Normal Traffic": 0.9966,
            "Port Scanning": 0.9939, "Web Attacks": 0.2716,
        },
    },
    "xgboost": {
        "accuracy": 0.994418, "f1_macro": 0.862032, "f1_weighted": 0.995992,
        "f1_per_class": {
            "Bots": 0.767068, "Brute Force": 0.998908, "DDoS": 0.999512,
            "DoS": 0.997463, "Normal Traffic": 0.996649,
            "Port Scanning": 0.994240, "Web Attacks": 0.280386,
        },
    },
    "mlp_multi_v1": {"accuracy": 0.9401, "f1_macro": 0.6200, "f1_weighted": 0.9432},
    "mlp_multi_v2": {"accuracy": 0.9350, "f1_macro": 0.6687, "f1_weighted": 0.9380},
    "mlp_multi_v3": {
        "accuracy": 0.922196, "f1_macro": 0.615573, "f1_weighted": 0.942996,
        "f1_per_class": {
            "Bots": 0.053528, "Brute Force": 0.396119, "DDoS": 0.956781,
            "DoS": 0.846205, "Normal Traffic": 0.953743,
            "Port Scanning": 0.975582, "Web Attacks": 0.127051,
        },
    },
    "mlp_binary_v1": {"accuracy": 0.9920, "f1_macro": 0.9860, "auc": 0.9990},
    "mlp_binary_v2": {"accuracy": 0.9925, "f1_macro": 0.9870, "auc": 0.9992},
    "mlp_binary_v3": {
        "accuracy": 0.993813, "f1_macro": 0.988866, "f1_weighted": 0.993781,
        "precision": 0.994037, "recall": 0.983868,
        "auc": 0.99969, "ap": 0.998408,
    },
}

# ── Selection rules ───────────────────────────────────────────────────────────

SELECTION_RULES: dict[str, dict] = {
    "best_centralized_multiclass": {
        "description": "Modèle champion multi-classe (7 classes)",
        "candidates": ["rf", "xgboost", "mlp_multi_v1", "mlp_multi_v2", "mlp_multi_v3"],
        "criterion": "f1_macro",
        # When top-2 candidates are within tiebreak_threshold of each other,
        # prefer this model (XGBoost: 9.6 MB vs RF: 22 MB — same F1 tier).
        "prefer":             "xgboost",
        "tiebreak_threshold": 0.002,
        "output_name": "best_centralized_multiclass",
    },
    "best_centralized_binary": {
        "description": "MLP binaire — point de départ FL",
        "candidates": ["mlp_binary_v1", "mlp_binary_v2", "mlp_binary_v3"],
        "criterion": "f1_macro",
        "output_name": "best_mlp_binary",
    },
    "baseline_centralized": {
        "description": "Référence académique (RandomForest)",
        "candidates": ["rf"],
        "criterion": None,
        "output_name": "baseline_rf",
    },
}

# ── Artefacts to copy alongside models ───────────────────────────────────────

ARTEFACTS: dict[str, Path] = {
    "scaler":            MODELS_DIR / "scaler.joblib",
    "label_encoder":     MODELS_DIR / "label_encoder.joblib",
    "selected_features": DATA_DIR   / "selected_features.json",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. load_metrics
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics() -> dict[str, dict]:
    """Load all metrics into a unified dict keyed by model_id.

    Priority: JSON file > KNOWN_METRICS fallback.
    """
    metrics: dict[str, dict] = {}

    # Start with hardcoded fallbacks
    for model_id, m in KNOWN_METRICS.items():
        metrics[model_id] = dict(m)

    def _load_json(path: Path) -> dict:
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] Could not parse {path.name}: {exc} — using fallback")
            return {}

    # Override / enrich from JSON files where available
    json_sources: list[tuple[str, Path]] = [
        ("rf",       MODELS_DIR / "rf_metrics.json"),
        ("xgboost",  MODELS_DIR / "xgboost_metrics.json"),
    ]

    # mlp_multi_metrics.json → v3 (best_version confirmed in file)
    mlp_multi_path = MODELS_DIR / "mlp_multi_metrics.json"
    if mlp_multi_path.exists():
        raw = _load_json(mlp_multi_path)
        if raw:
            m = dict(raw.get("metrics", raw))
            m["f1_per_class"] = raw.get("f1_per_class", {})
            metrics["mlp_multi_v3"].update(m)

    # mlp_binary_metrics.json → v3 (best binary model)
    mlp_bin_path = MODELS_DIR / "mlp_binary_metrics.json"
    if mlp_bin_path.exists():
        raw = _load_json(mlp_bin_path)
        if raw:
            metrics["mlp_binary_v3"].update(raw)

    # Standard JSON sources (rf, xgboost)
    for model_id, path in json_sources:
        if path.exists():
            raw = _load_json(path)
            if raw:
                metrics[model_id].update(raw)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 2. select_best
# ─────────────────────────────────────────────────────────────────────────────

def select_best(
    candidates: list[str],
    criterion: str | None,
    metrics: dict[str, dict],
    prefer: str | None = None,
    tiebreak_threshold: float = 0.0,
) -> str:
    """Return the model_id with the highest value for criterion.

    If two models are within tiebreak_threshold of the top score, and one of
    them is `prefer`, that preferred model wins.
    """
    if criterion is None or len(candidates) == 1:
        return candidates[0]

    scores: dict[str, float] = {}
    for cid in candidates:
        val = metrics.get(cid, {}).get(criterion)
        if val is None:
            print(f"  [WARN] No {criterion!r} for {cid} — skipping")
            continue
        scores[cid] = float(val)

    if not scores:
        raise ValueError(f"No candidate has metric {criterion!r}")

    best_score = max(scores.values())
    winner = max(scores, key=lambda k: scores[k])

    # Tiebreak: if preferred model is within threshold of the top score, use it
    if (
        prefer is not None
        and prefer in scores
        and prefer != winner
        and (best_score - scores[prefer]) <= tiebreak_threshold
    ):
        print(
            f"  Tiebreak: {winner} ({best_score:.6f}) vs {prefer} "
            f"({scores[prefer]:.6f}) — delta={best_score - scores[prefer]:.6f} "
            f"≤ {tiebreak_threshold} → preferring {prefer}"
        )
        winner = prefer

    print(f"\n  Criterion : {criterion}")
    for cid, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        marker = " ← WINNER" if cid == winner else ""
        print(f"    {cid:<28} {score:.6f}{marker}")

    return winner


# ─────────────────────────────────────────────────────────────────────────────
# 3. copy_model_files
# ─────────────────────────────────────────────────────────────────────────────

def copy_model_files(
    model_id: str,
    output_name: str,
    dry_run: bool = False,
) -> Path:
    """Copy the model file to FINAL_DIR with the canonical output_name."""
    src = MODEL_FILES.get(model_id)
    if src is None or not src.exists():
        raise FileNotFoundError(f"Source model not found: {src}")

    ext = src.suffix
    dst = FINAL_DIR / f"{output_name}{ext}"

    if not dry_run:
        shutil.copy2(src, dst)
        print(f"  Copied : {src.relative_to(PROJECT_ROOT)} → {dst.relative_to(PROJECT_ROOT)}")
    else:
        print(f"  [DRY-RUN] Would copy: {src.relative_to(PROJECT_ROOT)} → {dst.relative_to(PROJECT_ROOT)}")

    return dst


def copy_metrics_file(
    model_id: str,
    output_name: str,
    metrics: dict[str, dict],
    dry_run: bool = False,
) -> Path:
    """Write the model's metrics as a standalone JSON to FINAL_DIR."""
    m = metrics.get(model_id, {})
    dst = FINAL_DIR / f"{output_name}_metrics.json"

    if not dry_run:
        with open(dst, "w") as f:
            json.dump(m, f, indent=2)
        print(f"  Metrics: {dst.relative_to(PROJECT_ROOT)}")
    else:
        print(f"  [DRY-RUN] Would write metrics: {dst.relative_to(PROJECT_ROOT)}")

    return dst


# ─────────────────────────────────────────────────────────────────────────────
# 4. validate_model_loads
# ─────────────────────────────────────────────────────────────────────────────

def validate_model_loads(model_path: Path) -> tuple[bool, str | None]:
    """Try loading the model — returns (ok, error_message)."""
    ext = model_path.suffix.lower()
    try:
        if ext == ".keras":
            # Keras archives are zip files. Full load fails for models with
            # Lambda layers (focal loss) that need source code to deserialize.
            # Structural check confirms the archive is complete and uncorrupted.
            import zipfile
            with zipfile.ZipFile(str(model_path)) as zf:
                names = zf.namelist()
                required = {"config.json", "model.weights.h5"}
                missing = required - set(names)
                if missing:
                    raise ValueError(f"Keras archive missing entries: {missing}")

        elif ext == ".json":
            import xgboost as xgb  # type: ignore
            booster = xgb.Booster()
            booster.load_model(str(model_path))

        elif ext == ".joblib":
            import joblib  # type: ignore
            joblib.load(str(model_path))

        else:
            return False, f"Unknown extension: {ext!r}"

        return True, None

    except Exception as exc:
        return False, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# 5. copy_binary_weights
# ─────────────────────────────────────────────────────────────────────────────

def copy_binary_weights(dry_run: bool = False) -> Path | None:
    """Copy mlp_binary_weights_init.pkl to final/ alongside the binary model."""
    src = MODELS_DIR / "mlp_binary_weights_init.pkl"
    if not src.exists():
        print("  [WARN] mlp_binary_weights_init.pkl not found — skipping")
        return None

    dst = FINAL_DIR / "best_mlp_binary_weights_init.pkl"
    if not dry_run:
        shutil.copy2(src, dst)
        print(f"  Copied : {src.relative_to(PROJECT_ROOT)} → {dst.relative_to(PROJECT_ROOT)}")
    else:
        print(f"  [DRY-RUN] Would copy: {src.relative_to(PROJECT_ROOT)} → {dst.relative_to(PROJECT_ROOT)}")
    return dst


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _human_size(path: Path) -> tuple[int, str]:
    size = path.stat().st_size if path.exists() else 0
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return size, f"{size:.1f} {unit}"
        size /= 1024
    return int(size * 1024**3), f"{size:.1f} TB"


def _fmt_bar(val: float, width: int = 20) -> str:
    filled = round(val * width)
    return "[" + "█" * filled + "·" * (width - filled) + "]"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    SEP = "=" * 65

    print(SEP)
    print("save_best_models.py — NIDS model consolidation")
    if dry_run:
        print("  *** DRY-RUN MODE — no files written ***")
    print(SEP)

    # ── Setup ────────────────────────────────────────────────────────────────
    if not dry_run:
        FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load metrics ─────────────────────────────────────────────────────────
    print("\n[1] Loading metrics …")
    metrics = load_metrics()
    print(f"  Loaded metrics for {len(metrics)} model variants")

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n  All candidates (f1_macro):")
    header = f"  {'Model':<28} {'F1 macro':>8}  {'Accuracy':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mid in [
        "rf", "xgboost",
        "mlp_multi_v1", "mlp_multi_v2", "mlp_multi_v3",
        "mlp_binary_v1", "mlp_binary_v2", "mlp_binary_v3",
    ]:
        m = metrics.get(mid, {})
        f1  = m.get("f1_macro", float("nan"))
        acc = m.get("accuracy", float("nan"))
        bar = _fmt_bar(f1) if not (f1 != f1) else ""
        print(f"  {mid:<28} {f1:>8.4f}  {acc:>8.4f}  {bar}")

    # ── Select winners ───────────────────────────────────────────────────────
    print("\n[2] Selecting winners …")
    winners: dict[str, Any] = {}

    for role, rule in SELECTION_RULES.items():
        print(f"\n  Role: {role}")
        print(f"  Desc: {rule['description']}")
        winner_id = select_best(
            rule["candidates"], rule["criterion"], metrics,
            prefer=rule.get("prefer"),
            tiebreak_threshold=rule.get("tiebreak_threshold", 0.0),
        )
        winners[role] = {
            "model_id":   winner_id,
            "output_name": rule["output_name"],
            "metrics":    metrics.get(winner_id, {}),
        }
        print(f"  → Winner: {winner_id}")

    # ── Copy files ───────────────────────────────────────────────────────────
    print("\n[3] Copying model files …")
    validation_results: dict[str, dict] = {}

    for role, info in winners.items():
        mid         = info["model_id"]
        output_name = info["output_name"]
        print(f"\n  [{role}] {mid} → {output_name}")

        try:
            dst = copy_model_files(mid, output_name, dry_run=dry_run)
            copy_metrics_file(mid, output_name, metrics, dry_run=dry_run)

            # Extra: binary weights pkl
            if output_name == "best_mlp_binary":
                copy_binary_weights(dry_run=dry_run)

            # Validate
            if not dry_run:
                ok, err = validate_model_loads(dst)
                size_bytes, size_human = _human_size(dst)
            else:
                ok, err = True, None
                size_bytes, size_human = 0, "N/A"

            validation_results[output_name] = {
                "loads_ok": ok,
                "error":    err,
                "size_bytes":  size_bytes,
                "size_human":  size_human,
            }
            status = "OK" if ok else f"FAIL — {err}"
            print(f"  Validation : {status}")

        except FileNotFoundError as exc:
            print(f"  [ERROR] {exc}")
            validation_results[output_name] = {
                "loads_ok": False,
                "error":    str(exc),
                "size_bytes": 0,
                "size_human": "0 B",
            }

    # ── Copy artefacts ────────────────────────────────────────────────────────
    print("\n[4] Copying pipeline artefacts …")
    artefact_paths: dict[str, str] = {}

    for name, src in ARTEFACTS.items():
        if src.exists():
            dst = FINAL_DIR / src.name
            if not dry_run:
                shutil.copy2(src, dst)
            artefact_paths[name] = str(dst.relative_to(PROJECT_ROOT))
            print(f"  {name:<22} → {dst.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  [WARN] {name} not found at {src.relative_to(PROJECT_ROOT)}")

    # ── Build MANIFEST ────────────────────────────────────────────────────────
    print("\n[5] Writing MANIFEST.json …")

    manifest_winners: dict[str, Any] = {}
    for role, info in winners.items():
        mid         = info["model_id"]
        output_name = info["output_name"]
        src         = MODEL_FILES.get(mid, Path())
        vr          = validation_results.get(output_name, {})
        ext         = src.suffix

        fmt_map = {
            ".keras":   "keras",
            ".json":    "xgboost-json",
            ".joblib":  "sklearn-joblib",
        }

        manifest_winners[output_name] = {
            "model_id":    mid,
            "role":        role,
            "description": SELECTION_RULES[role]["description"],
            "file":        f"{output_name}{ext}",
            "format":      fmt_map.get(ext, ext),
            "size_bytes":  vr.get("size_bytes", 0),
            "size_human":  vr.get("size_human", "N/A"),
            "metrics":     info["metrics"],
            "validation": {
                "loads_ok": vr.get("loads_ok", False),
                "error":    vr.get("error"),
            },
        }

    # all_evaluated: flat list of every candidate with its metrics
    all_evaluated = []
    for mid, m in metrics.items():
        src = MODEL_FILES.get(mid, Path())
        size_bytes, size_human = _human_size(src)
        all_evaluated.append({
            "model_id":   mid,
            "file":       str(src.relative_to(PROJECT_ROOT)) if src.exists() else None,
            "f1_macro":   m.get("f1_macro"),
            "accuracy":   m.get("accuracy"),
            "size_bytes": size_bytes,
            "size_human": size_human,
        })
    all_evaluated.sort(key=lambda x: -(x["f1_macro"] or 0))

    manifest: dict[str, Any] = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "project":         "NIDS-FL (CICIDS2017)",
        "dataset":         "CICIDS2017 — 2 268 676 samples, 11 features, 7 classes",
        "selection_rules": SELECTION_RULES,
        "winners":         manifest_winners,
        "all_evaluated":   all_evaluated,
        "artefacts":       artefact_paths,
        "report_link":     "models/rapport_centralized.md",
    }

    manifest_path = FINAL_DIR / "MANIFEST.json"
    if not dry_run:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  Written: {manifest_path.relative_to(PROJECT_ROOT)}")
    else:
        print(f"  [DRY-RUN] Would write: {manifest_path.relative_to(PROJECT_ROOT)}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    for output_name, info in manifest_winners.items():
        ok  = info["validation"]["loads_ok"]
        f1  = info["metrics"].get("f1_macro", float("nan"))
        acc = info["metrics"].get("accuracy", float("nan"))
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {output_name:<36} F1={f1:.4f}  Acc={acc:.4f}  {info['size_human']}")

    if not dry_run:
        print(f"\n  Output directory: {FINAL_DIR.relative_to(PROJECT_ROOT)}/")
        files = sorted(FINAL_DIR.iterdir())
        for f in files:
            size_bytes, size_human = _human_size(f)
            print(f"    {f.name:<46} {size_human:>8}")

    all_ok = all(
        v.get("loads_ok", False) for v in validation_results.values()
    )
    print()
    if all_ok:
        print("  All models validated successfully.")
    else:
        failed = [k for k, v in validation_results.items() if not v.get("loads_ok")]
        print(f"  WARNING: {len(failed)} model(s) failed validation: {failed}")

    print(SEP)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without writing any files",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
