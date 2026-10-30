# Rapport — Modèles Centralisés NIDS (CICIDS2017)

**Date** : 2026-05-10  
**Dataset** : CICIDS2017 — 2 268 676 samples, 11 features, 7 classes  
**Split** : 70% train (1 764 525) / 10% val (252 075) / 20% test (504 151)

---

## Résultats comparatifs (test set)

| Modèle | Accuracy | F1 macro | F1 Bots | F1 Brute Force | F1 DDoS | F1 DoS | F1 Web Attacks | Taille |
|---|---|---|---|---|---|---|---|---|
| **XGBoost (Optuna)** | **99.44%** | **0.8620** | 0.767 | 0.999 | 0.9995 | 0.9975 | 0.280 | 9.6 MB |
| RandomForest | 99.43% | 0.8627 | 0.780 | 0.999 | 0.9998 | 0.9976 | 0.272 | 22 MB |
| MLP multi v3 (best) | 92.22% | 0.6156 | 0.054 | 0.396 | 0.957 | 0.846 | 0.127 | 1.1 MB |
| **MLP binaire** | **99.38%** | **0.9889** | — | — | — | — | — | 1.1 MB |

> Le MLP binaire est évalué sur la tâche Normal vs Attack (2 classes), pas comparable
> directement aux métriques 7 classes.

---

## Décision : Meilleurs modèles retenus

### 🏆 Champion centralisé (multi-classe) → XGBoost

**Fichier** : `models/xgboost_final.json`

**Pourquoi XGBoost plutôt que RF ?**
- F1 macro quasi identique (0.8620 vs 0.8627 RF) — différence non significative
- Modèle 2.3× plus léger (9.6 MB vs 22 MB)
- Inference plus rapide (arbres moins profonds, histogramme)
- Hyperparamètres Optuna-tuned (30 trials, TPE sampler)
- Meilleure accuracy globale (99.44% vs 99.43%)

**Paramètres retenus (Optuna)** :  
n_estimators=248, max_depth=10, lr=0.117, subsample=0.824,  
colsample_bytree=0.866, min_child_weight=2, gamma=0.006

### 🤖 Point de départ FL (binaire) → MLP Binary

**Fichier** : `models/mlp_binary.keras`  
**Poids FL** : `models/mlp_binary_weights_init.pkl`

**Métriques binaires** (Normal vs Attack) :
- Accuracy : 99.38%
- F1 macro : 0.9889
- Precision : 0.9940
- Recall : 0.9839
- AUC ROC : 0.9997
- Average Precision : 0.9984

**Pourquoi MLP binary pour le FL ?**
1. FedAvg agrège des listes de numpy arrays — incompatible avec arbres/boosting
2. Compact (1.1 MB, ~50k paramètres) — payload réseau négligeable par client
3. F1 binaire = 0.9889 → niveau production, convergence FL rapide attendue
4. La tâche binaire est plus robuste aux distributions locales de chaque client FL
5. Les poids centralisés (`warm-start`) remplaceront l'initialisation aléatoire

---

## Analyse : Pourquoi le MLP multi-classe n'atteint pas la cible (F1 = 0.6156 < 0.85)

### Classe Bots : F1 = 0.054 ← critique

- **195 samples de test** sur 504 151 (0.039%)
- Même avec class_weight × 200 et focal loss, le gradient reste dominé par les
  1.46M samples Normal Traffic
- Un réseau à 200k+ paramètres ne peut pas mémoriser 195 patterns sans sur-apprentissage

### Classe Web Attacks : F1 = 0.127

- 214 samples de test, patterns similaires à Normal Traffic (HTTP légitime vs attaque)
- Features réseau (11 colonnes) insuffisantes pour discriminer sans payload applicatif

### Comparaison vs RF/XGBoost

Les arbres de décision gèrent naturellement ces cas grâce à :
- **Partitions binaires dures** sur les features discrètes (flags TCP = 0/1)
- **Pas de gradient** → pas de domination des classes majoritaires
- **class_weight** implémenté nativement dans les critères de split

### Ce qui pourrait améliorer le MLP multi-classe

| Technique | Gain attendu | Complexité |
|---|---|---|
| SMOTE/ADASYN sur Bots (oversampling) | +0.10-0.20 F1 Bots | Moyenne |
| Architecture résiduelle (skip connections) | +0.05-0.10 F1 macro | Haute |
| Tuning Optuna du MLP (lr, dropout, units) | +0.05-0.15 F1 macro | Haute |
| Embeddings features (Destination Port) | +0.03-0.08 | Moyenne |

---

## Fichiers finaux

```
models/
├── xgboost_final.json          ← Champion centralisé (9.6 MB)
├── xgboost_metrics.json        ← F1 macro 0.8620, Acc 99.44%
├── mlp_binary.keras            ← Point départ FL (1.1 MB)
├── mlp_binary_weights_init.pkl ← Poids numpy pour FedAvg init
├── mlp_binary_metrics.json     ← F1 0.9889, AUC 0.9997
├── rf_metrics.json             ← Référence RF (F1 0.8627)
└── rapport_centralized.md      ← Ce fichier
```

---

## Prochaine étape

> **`notebooks/05_federated_learning.ipynb`**  
> Simulation FedAvg avec 3 clients, initialisation depuis `mlp_binary_weights_init.pkl`.  
> Cible : F1 binaire fédéré ≥ 0.95 après 20 rounds de communication.
