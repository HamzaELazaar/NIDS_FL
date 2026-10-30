# Dataset and models

Every figure on this page comes from files committed in the repository —
[`models/final/MANIFEST.json`](../models/final/MANIFEST.json),
[`models/final/selected_features.json`](../models/final/selected_features.json), and the
`*_metrics.json` files beside them.

## Dataset

**CICIDS2017**, cleaned and preprocessed variant
(`ericanacletoribeiro/cicids2017-cleaned-and-preprocessed`).

| Property | Value |
|---|---|
| Rows | 2,520,751 |
| Raw features | 52 |
| Selected features | 11 |
| Classes | 7 |
| Label column | `Attack Type` |

### Class distribution

The dataset is severely imbalanced, and this single fact explains most of the results below.

| Class | Count | Share |
|---|---|---|
| Normal Traffic | 2,095,057 | 83.11% |
| DoS | 193,745 | 7.69% |
| DDoS | 128,014 | 5.08% |
| Port Scanning | 90,694 | 3.60% |
| Brute Force | 9,150 | 0.36% |
| Web Attacks | 2,143 | 0.085% |
| Bots | 1,948 | 0.077% |

Web Attacks and Bots together account for **0.16%** of the data — roughly 4,000 rows out of 2.5
million.

### Selected features

Eleven features survived selection, benchmarked at accuracy 0.9973 / F1 macro 0.8658 with a
100-tree forest under 5-fold CV:

`Average Packet Size`, `Subflow Fwd Bytes`, `Destination Port`, `Init_Win_bytes_forward`,
`PSH Flag Count`, `FIN Flag Count`, `ACK Flag Count`, `Flow Bytes/s`, `Flow Packets/s`,
`Fwd Packets/s`, `Total Fwd Packets`

### Preprocessing

Strategy `clip_then_standard_scale`: clip each feature to its 1st–99th percentile bounds, then
standard-scale. The exact clip bounds are stored in `selected_features.json` and applied at
inference by the shared `FeatureScaler`, so live traffic is transformed identically to training
data. A few of the bounds:

| Feature | Low | High |
|---|---|---|
| `Average Packet Size` | 0.0 | 1454.125 |
| `Subflow Fwd Bytes` | 0.0 | 11,606.0 |
| `Destination Port` | 53.0 | 61,924.0 |
| `Init_Win_bytes_forward` | −1.0 | 65,535.0 |
| `Flow Bytes/s` | 0.0 | 12,333,333.33 |
| `Total Fwd Packets` | 1.0 | 51.0 |

Artefacts: `models/final/scaler.joblib`, `models/final/label_encoder.joblib`.

---

## Model registry

`MANIFEST.json` defines explicit **selection rules** rather than picking a winner by hand — each
role names its candidates, a criterion, and a tiebreak.

| Role | Candidates | Criterion | Winner |
|---|---|---|---|
| `best_centralized_multiclass` | rf, xgboost, mlp_multi_v1/v2/v3 | f1_macro, prefer xgboost, tiebreak 0.002 | **xgboost** |
| `best_centralized_binary` | mlp_binary_v1/v2/v3 | f1_macro | **mlp_binary_v3** |
| `baseline_centralized` | rf | — | **rf** |

### All evaluated models

| Model | F1 macro | Accuracy | Size |
|---|---|---|---|
| mlp_binary_v3 | 0.9889 | 0.9938 | 1.0 MB |
| mlp_binary_v2 | 0.9870 | 0.9925 | 1.0 MB |
| mlp_binary_v1 | 0.9860 | 0.9920 | 1.0 MB |
| rf | 0.8627 | 0.9943 | 21.6 MB |
| xgboost | 0.8620 | 0.9944 | 9.6 MB |
| mlp_multi_v2 (focal) | 0.6687 | 0.9350 | 1.0 MB |
| mlp_multi_v1 (balanced) | 0.6200 | 0.9401 | 1.0 MB |
| mlp_multi_v3 (combined) | 0.6156 | 0.9222 | 1.0 MB |

Binary and multiclass F1 scores are **not comparable** — the binary models solve an easier problem
(attack vs normal) and sit at the top of the table for that reason, not because they outperform the
7-class models at their own task.

---

## Champion — XGBoost + Optuna (7-class)

| Metric | Value |
|---|---|
| Accuracy | 0.994418 |
| F1 macro | 0.862032 |
| F1 weighted | 0.995992 |
| Train time | 454.55 s |
| Optuna search | 30 trials, 5,475.14 s |
| Best validation F1 | 0.863885 |
| Train / test | 2,016,600 / 504,151 |
| Artefact | `best_centralized_multiclass.json` (xgboost-json, 9.6 MB) |

Optuna-tuned hyperparameters:

```json
{
  "n_estimators": 248,
  "max_depth": 10,
  "learning_rate": 0.1171,
  "subsample": 0.8245,
  "colsample_bytree": 0.8661,
  "min_child_weight": 2,
  "gamma": 0.005687,
  "reg_alpha": 0.2936,
  "reg_lambda": 0.7509
}
```

### Per-class F1 — read this before quoting the headline number

| Class | F1 | Support share |
|---|---|---|
| DDoS | 0.9995 | 5.08% |
| Brute Force | 0.9989 | 0.36% |
| DoS | 0.9975 | 7.69% |
| Normal Traffic | 0.9966 | 83.11% |
| Port Scanning | 0.9942 | 3.60% |
| Bots | 0.7671 | 0.077% |
| Web Attacks | 0.2804 | 0.085% |

Accuracy of 99.4% is close to meaningless on this distribution — predicting "Normal Traffic" for
everything scores 83%. **F1 macro (0.862) is the number that matters**, and it is held down almost
entirely by Web Attacks (0.28) and Bots (0.77). Five of seven classes are detected essentially
perfectly; the two rarest are not, and no amount of tuning fixed that within this feature set.
Improving them is the clearest open problem in the project.

Optuna bought very little: the untuned baseline scored accuracy 0.99418 / F1 0.862068, so 5,475
seconds of search produced a **marginally lower** F1 macro than the baseline. That is recorded
honestly in the metrics file rather than hidden.

## Baseline — RandomForest

| Metric | Value |
|---|---|
| Accuracy | 0.994319 |
| F1 macro | 0.862657 |
| 5-fold CV F1 macro | 0.859285 ± 0.003525 |
| Config | 200 trees, max_depth 20, `class_weight="balanced"`, seed 42 |
| Train time | 255.68 s |
| Size | 21.6 MB |

CV scores: 0.8646, 0.8564, 0.8620, 0.8551, 0.8584.

RandomForest actually edges XGBoost on F1 macro (0.8627 vs 0.8620). XGBoost is selected because the
difference falls inside the manifest's 0.002 tiebreak band, and it is less than half the size —
which matters for a model shipped to inference nodes.

## Binary MLP — the federated model

| Metric | Value |
|---|---|
| Accuracy | 0.993813 |
| F1 macro | 0.988866 |
| F1 weighted | 0.993781 |
| Precision | 0.994037 |
| Recall | 0.983868 |
| ROC AUC | 0.999690 |
| Average precision | 0.998408 |
| Artefact | `best_mlp_binary.keras` (1.0 MB) |

Architecture 11 → 256 → 128 → 64 → 32 → 1 (sigmoid). This is the model that runs at the edge on
every packet and the one federated learning trains — see
[federated-learning.md](federated-learning.md).

Recall 0.9839 means roughly **1.6% of attacks are missed by the edge stage**. In the hybrid design
that is partially mitigated: uncertain packets are escalated regardless of predicted class, so only
*confidently misclassified* attacks are truly lost.

## Notebooks

| Notebook | Contents |
|---|---|
| `01_explore_dataset.ipynb` | EDA, class distribution, correlations, outliers |
| `02_random_forest.ipynb` | RF baseline, CV, feature importance |
| `03_xgboost_optuna.ipynb` | 30-trial Optuna search, SHAP |
| `04_mlp_centralized.ipynb` | MLP variants, binary and multiclass |
| `06_hybrid_vs_centralized.ipynb` | Hybrid pipeline vs single centralised model |

Generated figures live in [`images/`](images/) — confusion matrices, ROC/PR curves, feature
importances, SHAP summaries, Optuna history, and the hybrid latency and traffic-split plots. The
Optuna parallel-coordinates plot is an interactive HTML file (`images/xgb_optuna_parallel.html`).

A French write-up of the centralised phase is preserved at
[`../models/rapport_centralized.md`](../models/rapport_centralized.md).
