#!/usr/bin/env python3
"""
benchmark_features.py — Compare 7 vs 11 features (RandomForest, CICIDS2017)

Usage:
    python scripts/benchmark_features.py [--csv data/cicids2017_cleaned.csv]
                                         [--sample 100000]
                                         [--output data/benchmark_results.txt]
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# ─── Feature groups ──────────────────────────────────────────────────────────

FEATS_7 = [
    "Destination Port",
    "FIN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "Init_Win_bytes_forward",
    "Subflow Fwd Bytes",
    "Average Packet Size",
]

FEATS_11 = FEATS_7 + [
    "Total Fwd Packets",    # ← packet_count_1s
    "Flow Bytes/s",         # ← bytes_per_sec_1s
    "Flow Packets/s",       # ← packet_freq_1s
    "Fwd Packets/s",        # ← packet_freq_1s (alias)
]

TARGET = "Attack Type"

LABEL_MAP = {
    "BENIGN": "Normal Traffic",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "Port Scanning",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Bot": "Bots",
    "XSS": "Web Attacks",
    "Sql Injection": "Web Attacks",
    "Infiltration": "Web Attacks",
    "Web Attack – Brute Force": "Web Attacks",
    "Web Attack – XSS": "Web Attacks",
    "Web Attack – Sql Injection": "Web Attacks",
    "Heartbleed": "Web Attacks",
}

# ─── Terminal colours ─────────────────────────────────────────────────────────

BOLD  = "\033[1m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
YELLOW = "\033[33m"
RED   = "\033[31m"
DIM   = "\033[2m"
NC    = "\033[0m"


# ─── Formatting helpers ───────────────────────────────────────────────────────

def bar(value: float, max_value: float = 1.0, width: int = 20) -> str:
    filled = round(value / max_value * width) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def delta_str(a: float, b: float, pct: bool = False) -> str:
    diff = b - a
    if pct:
        diff *= 100
    sign = "+" if diff >= 0 else ""
    colour = GREEN if diff > 0 else (RED if diff < 0 else DIM)
    unit = " pp" if pct else ""
    return f"{colour}{sign}{diff:.2f}{unit}{NC}"


def section(title: str, char: str = "═", width: int = 66) -> str:
    pad = (width - len(title) - 2) // 2
    return f"\n{BOLD}{'═' * pad} {title} {'═' * (width - pad - len(title) - 2)}{NC}"


def print_and_save(text: str, fh):
    # strip ANSI for file
    import re
    plain = re.sub(r"\033\[[0-9;]*m", "", text)
    print(text)
    fh.write(plain + "\n")


# ─── Core ────────────────────────────────────────────────────────────────────

def load_data(csv_path: Path, sample: int) -> pd.DataFrame:
    print(f"\n{CYAN}⏳  Chargement de {csv_path.name} …{NC}")
    t0 = time.time()

    # Read only the columns we need
    needed = list(set(FEATS_11 + [TARGET]))
    try:
        df = pd.read_csv(csv_path, usecols=needed, low_memory=False)
    except ValueError:
        # Some cols may be absent — fall back to full load
        df = pd.read_csv(csv_path, low_memory=False)

    df.columns = df.columns.str.strip()
    df[TARGET] = df[TARGET].str.strip()

    # Normalise labels if raw BENIGN/etc.
    if df[TARGET].isin(LABEL_MAP.keys()).any():
        df[TARGET] = df[TARGET].map(LABEL_MAP).fillna(df[TARGET])

    # Drop rows with NaN in feature cols
    df = df.dropna(subset=[c for c in FEATS_11 if c in df.columns] + [TARGET])

    # Stratified sample
    if sample and len(df) > sample:
        df, _ = train_test_split(
            df, train_size=sample, stratify=df[TARGET], random_state=42
        )

    elapsed = time.time() - t0
    print(f"{GREEN}✅  {len(df):,} lignes chargées en {elapsed:.1f}s{NC}")
    return df


def train_evaluate(df: pd.DataFrame, features: list[str], label: str = "Groupe"):
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        print(f"{YELLOW}⚠️  Colonnes absentes ignorées: {missing}{NC}")

    X = df[available].fillna(0).values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred  = clf.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "label":       label,
        "features":    available,
        "accuracy":    acc,
        "report":      report,
        "importances": dict(zip(available, clf.feature_importances_)),
        "train_time":  train_time,
        "n_train":     len(X_train),
        "n_test":      len(X_test),
    }


def print_results(res_a: dict, res_b: dict, fh):
    classes = sorted(
        set(res_a["report"].keys()) & set(res_b["report"].keys()) - {"accuracy","macro avg","weighted avg"}
    )

    # ── Header ──────────────────────────────────────────────────────────────
    header = f"""
╔══════════════════════════════════════════════════════════════════╗
║        NIDS-FL — Benchmark Features  ({datetime.now():%Y-%m-%d %H:%M})          ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print_and_save(header, fh)

    # ── Setup ────────────────────────────────────────────────────────────────
    setup = (
        f"  Groupe A ({res_a['label']}) : {len(res_a['features'])} features\n"
        f"  Groupe B ({res_b['label']}) : {len(res_b['features'])} features\n"
        f"  Train / Test split     : 80 / 20\n"
        f"  Modèle                 : RandomForest (n_estimators=100)\n"
        f"  Train A / B            : {res_a['train_time']:.1f}s / {res_b['train_time']:.1f}s\n"
        f"  Échantillon            : {res_a['n_train'] + res_a['n_test']:,} lignes\n"
    )
    print_and_save(setup, fh)

    # ── Global accuracy ──────────────────────────────────────────────────────
    print_and_save(section("PRÉCISION GLOBALE"), fh)
    acc_a = res_a["accuracy"] * 100
    acc_b = res_b["accuracy"] * 100
    row = (
        f"\n  {'Groupe':<20} {'Accuracy':>10}   {'Delta':>10}\n"
        f"  {'─'*20} {'─'*10}   {'─'*10}\n"
        f"  {res_a['label']:<20} {acc_a:>9.2f}%\n"
        f"  {res_b['label']:<20} {acc_b:>9.2f}%   {delta_str(acc_a/100, acc_b/100, pct=True)}\n"
    )
    print_and_save(row, fh)

    # ── Per-class F1 ─────────────────────────────────────────────────────────
    print_and_save(section("F1-SCORE PAR CLASSE"), fh)
    hdr = f"\n  {'Classe':<26} {'F1-A':>7}  {'F1-B':>7}  {'Δ F1':>10}  {'Δ Recall':>10}\n"
    hdr += f"  {'─'*26} {'─'*7}  {'─'*7}  {'─'*10}  {'─'*10}"
    print_and_save(hdr, fh)

    for cls in classes:
        ra = res_a["report"].get(cls, {})
        rb = res_b["report"].get(cls, {})
        f1a = ra.get("f1-score", 0.0)
        f1b = rb.get("f1-score", 0.0)
        rca = ra.get("recall",   0.0)
        rcb = rb.get("recall",   0.0)
        sup = rb.get("support",  ra.get("support", 0))
        row = (
            f"  {cls:<26} {f1a:>7.3f}  {f1b:>7.3f}"
            f"  {delta_str(f1a, f1b, pct=True):>10}"
            f"  {delta_str(rca, rcb, pct=True):>10}"
            f"  (n={int(sup):,})"
        )
        print_and_save(row, fh)

    print_and_save("", fh)

    # ── Feature importance — Groupe A ────────────────────────────────────────
    print_and_save(section(f"IMPORTANCE DES FEATURES — {res_a['label']}"), fh)
    imp_a = sorted(res_a["importances"].items(), key=lambda x: x[1], reverse=True)
    max_a = imp_a[0][1] if imp_a else 1.0
    print_and_save("", fh)
    for feat, imp in imp_a:
        line = f"  {feat:<30} {imp:>6.3f}  {bar(imp, max_a)}"
        print_and_save(line, fh)

    # ── Feature importance — Groupe B ────────────────────────────────────────
    print_and_save(section(f"IMPORTANCE DES FEATURES — {res_b['label']}"), fh)
    imp_b = sorted(res_b["importances"].items(), key=lambda x: x[1], reverse=True)
    max_b = imp_b[0][1] if imp_b else 1.0

    # Tag new features
    new_feats = set(res_b["features"]) - set(res_a["features"])
    print_and_save("", fh)
    for feat, imp in imp_b:
        tag  = f"  {CYAN}← nouveau{NC}" if feat in new_feats else ""
        line = f"  {feat:<30} {imp:>6.3f}  {bar(imp, max_b)}{tag}"
        print_and_save(line, fh)

    # ── Pseudo-flow contribution ──────────────────────────────────────────────
    new_imp_sum = sum(v for k, v in res_b["importances"].items() if k in new_feats)
    print_and_save(section("CONTRIBUTION DES PSEUDO-FLOW FEATURES"), fh)
    contribution = (
        f"\n  Features pseudo-flow : {', '.join(sorted(new_feats))}\n"
        f"  Importance totale    : {new_imp_sum * 100:.1f}% du modèle B\n"
    )
    print_and_save(contribution, fh)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_and_save(section("RÉSUMÉ"), fh)
    summary = f"""
  ✅ Accuracy :  {acc_a:.2f}%  →  {acc_b:.2f}%   ({'+' if acc_b >= acc_a else ''}{acc_b - acc_a:.2f} pp)
  ✅ Pseudo-flow features représentent {new_imp_sum*100:.0f}% de l'importance totale (Groupe B)

  Gains notables par classe :"""
    print_and_save(summary, fh)

    for cls in classes:
        ra = res_a["report"].get(cls, {})
        rb = res_b["report"].get(cls, {})
        f1a = ra.get("f1-score", 0.0)
        f1b = rb.get("f1-score", 0.0)
        diff = (f1b - f1a) * 100
        if abs(diff) >= 0.5:
            arrow = "▲" if diff > 0 else "▼"
            colour = GREEN if diff > 0 else RED
            print_and_save(
                f"    {colour}{arrow} {cls:<24} F1 {f1a:.3f} → {f1b:.3f}  ({'+' if diff>=0 else ''}{diff:.1f} pp){NC}",
                fh,
            )

    print_and_save("\n" + "═" * 66 + "\n", fh)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NIDS-FL feature benchmark")
    parser.add_argument(
        "--csv",
        default="data/cicids2017_cleaned.csv",
        help="Chemin vers le CSV CICIDS2017",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100_000,
        help="Taille de l'échantillon stratifié (0 = tout)",
    )
    parser.add_argument(
        "--output",
        default="data/benchmark_results.txt",
        help="Fichier de sortie (texte brut sans ANSI)",
    )
    parser.add_argument(
        "--json",
        default="data/benchmark_results.json",
        help="Fichier de sortie JSON (métriques brutes)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"{RED}❌  Fichier introuvable : {csv_path}{NC}")
        sys.exit(1)

    out_path  = Path(args.output)
    json_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path, args.sample)

    print(f"\n{CYAN}🔬  Entraînement Groupe A — 7 features …{NC}")
    res_a = train_evaluate(df, FEATS_7,  label="7 features (base)")

    print(f"{CYAN}🔬  Entraînement Groupe B — 11 features …{NC}")
    res_b = train_evaluate(df, FEATS_11, label="11 features (+pseudo-flow)")

    with open(out_path, "w", encoding="utf-8") as fh:
        print_results(res_a, res_b, fh)

    # Save raw metrics as JSON
    def _serialise(r: dict) -> dict:
        return {
            "label":      r["label"],
            "features":   r["features"],
            "accuracy":   round(r["accuracy"], 6),
            "train_time": round(r["train_time"], 2),
            "n_train":    r["n_train"],
            "n_test":     r["n_test"],
            "importances": {k: round(v, 6) for k, v in r["importances"].items()},
            "report": {
                cls: (
                    {m: round(v, 4) for m, v in metrics.items()}
                    if isinstance(metrics, dict)
                    else round(metrics, 4)
                )
                for cls, metrics in r["report"].items()
            },
        }

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(
            {
                "generated": datetime.now().isoformat(),
                "sample":    args.sample,
                "group_a":   _serialise(res_a),
                "group_b":   _serialise(res_b),
            },
            jf,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{GREEN}📄  Résultats texte → {out_path}{NC}")
    print(f"{GREEN}📊  Résultats JSON  → {json_path}{NC}\n")


if __name__ == "__main__":
    main()
