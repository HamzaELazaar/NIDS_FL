# NIDS-FL — Full System Flow

Network Intrusion Detection System with Federated Learning on CICIDS2017.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Dataset](#2-dataset)
3. [Phase 1 — Data Exploration & Preprocessing](#3-phase-1--data-exploration--preprocessing)
4. [Phase 2 — Centralized Model Training](#4-phase-2--centralized-model-training)
5. [Phase 3 — Federated Learning](#5-phase-3--federated-learning)
6. [Phase 4 — Hybrid Architecture](#6-phase-4--hybrid-architecture)
7. [Phase 5 — Live Inference API](#7-phase-5--live-inference-api)
8. [Phase 6 — Kafka Streaming Pipeline](#8-phase-6--kafka-streaming-pipeline)
9. [Infrastructure](#9-infrastructure)
10. [Model Performance Summary](#10-model-performance-summary)
11. [Project File Structure](#11-project-file-structure)
12. [Running the Full Stack](#12-running-the-full-stack)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NIDS-FL PIPELINE                            │
│                                                                     │
│  CICIDS2017 CSV (685 MB)                                            │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐   Feature     ┌─────────────┐   11 features       │
│  │  EDA &      │   selection   │ Preprocessing│──────────────────┐  │
│  │  Baseline   │──────────────▶│ + Scaler     │                  │  │
│  │  (RF)       │               │ (StandardScl)│                  │  │
│  └─────────────┘               └─────────────┘                  │  │
│         │                                                        │  │
│         ▼  Centralized Training                                  │  │
│  ┌──────────────────────────────────────────┐                   │  │
│  │   XGBoost (Optuna, 30 trials)            │                   │  │
│  │   MLP binary  v1/v2/v3  (Keras)          │                   │  │
│  │   MLP multiclass v1/v2/v3                │                   │  │
│  └──────────────────────────────────────────┘                   │  │
│         │                                                        │  │
│         ▼  Federated Learning (FedAvg)                          │  │
│  ┌──────────────────────────────────────────┐                   │  │
│  │  FL Server  ←──── Kafka ────▶ FL Client  │                   │  │
│  │  (FedAvg)         ×3         (local MLP) │                   │  │
│  └──────────────────────────────────────────┘                   │  │
│         │                                                        │  │
│         ▼  Hybrid Inference (no Kafka needed)                   │  │
│  ┌────────────────────────────────────┐                         │  │
│  │ Edge Detector  (binary MLP 64→32→16│◀────────────────────────┘  │
│  │ threshold = 0.95                   │                             │
│  │  ├─ Normal (conf ≥ 0.95) → DROP   │                             │
│  │  └─ Attack / uncertain  → Cloud   │                             │
│  └────────────────────────────────────┘                             │
│         │  ~20.7 % of traffic                                       │
│         ▼                                                           │
│  ┌────────────────────────────────────┐                             │
│  │ Cloud Analyzer (XGBoost 7-class)  │                             │
│  │ 7 classes: Normal/DDoS/DoS/Bots/  │                             │
│  │ Brute Force/Port Scan/Web Attacks  │                             │
│  └────────────────────────────────────┘                             │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────────────────┐                             │
│  │  FastAPI (port 8001)               │                             │
│  │  POST /predict  GET /alerts        │                             │
│  │  GET /stats     WS  /stream        │                             │
│  └────────────────────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset

**Source:** CICIDS2017 (Canadian Institute for Cybersecurity)  
**File:** `data/cicids2017_cleaned.csv` — 685 MB, 2,520,751 rows, 53 columns

### Label Distribution

| Class | Count | % |
|---|---|---|
| Normal Traffic | 2,095,057 | 83.11% |
| DoS | 193,745 | 7.69% |
| DDoS | 128,014 | 5.08% |
| Port Scanning | 90,694 | 3.60% |
| Brute Force | 9,150 | 0.36% |
| Web Attacks | 2,143 | 0.09% |
| Bots | 1,948 | 0.08% |

### Train / Val / Test Split

```
Total filtered: 2,268,676 samples (after cleaning)
│
├── Train  : 1,764,525  (70 %)   → data/processed/train.npz
├── Val    :   252,075  (10 %)   → data/processed/val.npz
└── Test   :   504,151  (20 %)   → data/processed/test.npz

Federated splits (3 clients, IID by default):
  └── data/processed/fl_client_{1,2,3}.npz
```

### 11 Selected Features (used by all ML models)

| # | CICIDS2017 Column | Live API Mapping |
|---|---|---|
| 1 | Average Packet Size | `avg_packet_size_1s` (rolling 1s window) |
| 2 | Subflow Fwd Bytes | `payload_length` |
| 3 | Destination Port | `dst_port` |
| 4 | Init_Win_bytes_forward | `window_size` |
| 5 | PSH Flag Count | `psh_flag` |
| 6 | FIN Flag Count | `fin_flag` |
| 7 | ACK Flag Count | `ack_flag` |
| 8 | Flow Bytes/s | `bytes_per_sec_1s` (rolling 1s) |
| 9 | Flow Packets/s | `packet_freq_1s` (rolling 1s) |
| 10 | Fwd Packets/s | `packet_freq_1s` (same approximation) |
| 11 | Total Fwd Packets | `packet_count_1s` (rolling 1s) |

---

## 3. Phase 1 — Data Exploration & Preprocessing

**Notebooks:** `01_explore_dataset.ipynb`, `02_random_forest.ipynb`

### Steps

```
CSV load (pandas)
    │
    ▼
Drop duplicates, handle Inf/NaN, clip outliers (IQR ×3)
    │
    ▼
Feature selection (11 out of 53 CICIDS2017 features)
    │  Selected via correlation analysis + RF importance + CICIDS→live mappability
    ▼
StandardScaler fit on train set only
    │  Saved to: models/final/scaler.joblib
    ▼
LabelEncoder for 7-class labels
    │  Saved to: models/final/label_encoder.joblib
    ▼
Save splits:  data/processed/{train,val,test,fl_client_1,2,3}.npz
    │  Keys: X (scaled), y_multi (0–6), y_binary (0=Normal, 1=Attack)
    ▼
RandomForest baseline (sklearn)
    │  n_estimators=200, max_depth=20, class_weight='balanced'
    │  5-fold CV F1-macro = 0.859 ± 0.004
    └  → models/final/baseline_rf.joblib
```

### Preprocessing module

`federated_learning/preprocessing.py` — `FeatureScaler` class

```python
# Clip + StandardScale in one step
scaler = FeatureScaler.load("models/final")
X_scaled = scaler.transform(X_raw)   # used at inference time
```

---

## 4. Phase 2 — Centralized Model Training

### 4.1 XGBoost — Champion Multiclass Model

**Notebook:** `03_xgboost_optuna.ipynb`

```
Optuna hyperparameter search
    │  30 trials, objective = val F1-macro
    │  Search space: n_estimators, max_depth, learning_rate,
    │                subsample, colsample_bytree, gamma, reg_alpha, reg_lambda
    ▼
Best params found:
    n_estimators=248, max_depth=10, learning_rate=0.117,
    subsample=0.824, colsample_bytree=0.866
    ▼
Final training on full train set (1,764,525 samples)
    │  train_time = 455 s | optuna_time = 5,475 s
    ▼
Saved: models/final/best_centralized_multiclass.json  (9.6 MB)
```

**Test set results:**

| Metric | Value |
|---|---|
| Accuracy | 99.44% |
| F1 macro | 0.862 |
| F1 weighted | 0.996 |
| F1 Brute Force | 0.9989 |
| F1 DDoS | 0.9995 |
| F1 DoS | 0.9975 |
| F1 Port Scanning | 0.9942 |
| F1 Bots | 0.767 |
| F1 Web Attacks | 0.280 (minority class) |

### 4.2 Binary MLP — Federated Learning Starting Point

**Notebook:** `04_mlp_centralized.ipynb`

Three versions trained with increasing sophistication:

```
v1 — Standard cross-entropy + class weights
v2 — Focal loss (γ=2) + SMOTE oversampling of minority classes
v3 — Combined: focal loss + class weights + cosine annealing LR
    │
    ▼  EarlyStopping checkpoint saved at best val F1
    │
    └─ Winner: v3 (mlp_binary_ckpt.keras)
```

**Architecture (v3 checkpoint):**

```
Input(11)
  → Dense(64, relu) → BatchNorm → Dropout(0.3)
  → Dense(32, relu) → BatchNorm → Dropout(0.3)
  → Dense(16, relu) → Dropout(0.2)
  → Dense(1, sigmoid)
Total params: 3,777
```

**Best binary model metrics:**

| Metric | Value |
|---|---|
| Accuracy | 99.38% |
| F1 macro | 0.9687 |
| AUC | 0.9990 |
| Attack Recall | 99.5% |
| Precision | 99.4% |

Saved to: `models/mlp_binary_ckpt.keras` (1.0 MB)

---

## 5. Phase 3 — Federated Learning

**Architecture:** FedAvg (McMahan et al., 2017) over Kafka

```
┌──────────────────────────────────────────────────────────────────┐
│  Federated Learning Round (repeated n_rounds times)              │
│                                                                  │
│  FL Server                      FL Client ×3                     │
│  ─────────────                  ─────────────────────────────    │
│  1. Broadcast global weights ──▶ Receive weights (global-model)  │
│                                  Update local model              │
│                                  Train local_epochs on own data  │
│                                  Evaluate on local val set       │
│                                  Serialize & compress weights    │
│  2. Receive all client weights ◀─ Publish to 'model-weights'     │
│  3. FedAvg aggregation           n_i                             │
│     w_global = Σ ─────── × w_i                                   │
│                  N_total                                         │
│  4. Evaluate global model        (on server-held test set)       │
│  5. Persist & broadcast ────────▶ Receive new global weights     │
└──────────────────────────────────────────────────────────────────┘
```

### Kafka Topics (FL)

| Topic | Producer | Consumer | Content |
|---|---|---|---|
| `model-weights` | FL Client | FL Server | Serialized gzip+base64 weights |
| `global-model` | FL Server | FL Clients | Aggregated global weights |

### Weight serialization

`federated_learning/aggregation.py`:
```
numpy arrays → pickle → gzip compress → base64 encode → JSON → Kafka message
Size: ~15 KB/message (3,777 params, float32)
```

### Running FL

```bash
# Terminal 1 — Server
python -m federated_learning.fl_server --rounds 10 --clients 3

# Terminal 2,3,4 — Clients
python -m federated_learning.fl_client --id 1 --rounds 10
python -m federated_learning.fl_client --id 2 --rounds 10
python -m federated_learning.fl_client --id 3 --rounds 10
```

---

## 6. Phase 4 — Hybrid Architecture

**Notebook:** `06_hybrid_vs_centralized.ipynb`

### Concept

Rather than sending every packet to the heavy XGBoost model, the binary MLP acts as a **lightweight edge filter**: only traffic it cannot confidently classify as normal is forwarded to XGBoost for deep analysis.

```
Every packet
    │
    ▼
Binary MLP (edge)  [11 features, 3,777 params, ~0.25 ms/pkt]
    │
    ├─ P(Normal) ≥ 0.95  ─────────────▶  DROP  (bandwidth saved)
    │                                    79.3% of all traffic
    │
    └─ Attack confident                ─┐
       OR uncertain (P < 0.95 gate)    ─┴▶  XGBoost (cloud) [9.6 MB]
                                            7-class classification
                                            20.7% of all traffic
```

### Routing Decision Table

| Edge confidence | Edge prediction | Action |
|---|---|---|
| ≥ 0.95 | Normal | **DROP** — certain benign |
| ≥ 0.95 | Attack | **Forward** — certain attack |
| < 0.95 | Either | **Forward** — uncertain, needs cloud |

### Hybrid vs Centralized (XGBoost alone) — Test Set Results

| Metric | XGBoost alone (Arch A) | Hybrid MLP→XGBoost (Arch B) | Delta |
|---|---|---|---|
| Accuracy | 99.44% | 99.44% | −0.006 pp |
| F1 macro | 0.8620 | 0.8575 | −0.0046 |
| F1 weighted | 0.9960 | 0.9959 | −0.0001 |
| Latency (ms/pkt) | 0.0093 | 0.0096 | +3.3% |
| XGBoost calls / 1000 | 1,000 | **207** | **−79.3%** |
| Cloud traffic | 100% | **20.7%** | **−79.3%** |
| Edge FN rate | — | 0.07% | — |

**At threshold = 0.95: accuracy cost = 0.006 pp, cloud traffic reduction = 79.3%.**

### Hybrid pipeline components

| File | Role |
|---|---|
| `hybrid/edge_detector.py` | `EdgeDetector` — binary MLP stage, Kafka consumer/producer |
| `hybrid/cloud_analyzer.py` | `CloudAnalyzer` — XGBoost stage, Kafka consumer/producer |
| `hybrid/orchestrator.py` | `HybridOrchestrator` — consumes all 4 topics, computes metrics, saves `data/hybrid_metrics.json` every 10s, Prometheus endpoint on `:9100/metrics` |

### Kafka Topics (Hybrid)

| Topic | Producer | Consumer | Content |
|---|---|---|---|
| `network-events` | Kafka Bridge / API | Edge Detector | Raw 11-feature JSON per packet |
| `suspicious-traffic` | Edge Detector | Cloud Analyzer | Features + edge prediction metadata |
| `classified-attacks` | Cloud Analyzer | Orchestrator | Full enriched alert (all classified) |
| `alerts` | Cloud Analyzer | Orchestrator / Dashboard | Confirmed attacks only |

### Running the hybrid pipeline (Kafka mode)

```bash
# Terminal 1
python -m hybrid.edge_detector --threshold 0.95 --batch-size 32

# Terminal 2
python -m hybrid.cloud_analyzer --log-interval 50

# Terminal 3
python -m hybrid.orchestrator
```

---

## 7. Phase 5 — Live Inference API

**File:** `api/main.py`  
**Run:** `uvicorn api.main:app --host 0.0.0.0 --port 8001`

### Startup sequence

```
uvicorn starts
    │
    ▼
NIDSPipeline.load()  [in thread executor — non-blocking]
    ├── Load models/mlp_binary_ckpt.keras       (edge MLP)
    ├── Warmup: dummy predict (batch=1, input=zeros)
    ├── Load models/final/best_centralized_multiclass.json  (cloud XGBoost)
    ├── Load models/final/scaler.joblib         (StandardScaler)
    └── Load models/final/label_encoder.joblib  (7-class LabelEncoder)
    │
    ▼
capture_loop() starts  [async task on eth2]
    │
    ▼  Per packet (async):
    │
    ├── pyshark captures IP packet (TCP/UDP/ICMP)
    ├── extract_features(packet)  → 22-feature dict
    │   └── FlowTracker updates rolling 1s window per src_ip
    ├── normalize_features()      → cleaned scalars
    ├── to_ml_vector()            → 11-feature CICIDS2017 dict
    │
    └── run_in_executor(NIDSPipeline.classify)
            │
            ├── EdgeDetector.predict_packet()
            │   ├── scaler.transform(raw_11)
            │   └── mlp.predict(scaled)  → sigmoid proba
            │
            ├── if suspicious:
            │   └── CloudAnalyzer.classify()
            │       ├── scaler.transform(raw_11)
            │       ├── xgb.DMatrix(scaled)
            │       └── booster.predict()  → 7-class probas
            │
            └── append to PREDICTIONS (deque maxlen=2000)
                if severity != "none": append to ALERTS (deque maxlen=500)
                broadcast to WebSocket clients
```

### Feature extraction — live approximations

Since CICIDS2017 features are **flow-level** (session aggregates) and live capture is **packet-level**, 4 features are approximated via `FlowTracker`, a rolling 1-second window per source IP:

```
FlowTracker.update(src_ip, timestamp, packet_length)
    → packet_count_1s    ≈  Total Fwd Packets
    → bytes_per_sec_1s   ≈  Flow Bytes/s
    → avg_packet_size_1s ≈  Average Packet Size
    → packet_freq_1s     ≈  Flow Packets/s  AND  Fwd Packets/s
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Classify one packet. Accepts 11-key (CICIDS) or 22-key (live) feature dict |
| `GET` | `/predictions` | Recent classified packets. Filters: `label`, `severity`, `stage`, `limit` |
| `GET` | `/alerts` | Confirmed attacks only. Filter: `severity` |
| `GET` | `/packets` | Raw captured packets. Filters: `protocol`, `src_ip`, `dst_ip`, `dst_port` |
| `GET` | `/stats` | Protocol distribution, top IPs/ports, label counts, severity breakdown |
| `GET` | `/pipeline/status` | Edge model name, threshold, XGBoost classes |
| `GET` | `/health` | Liveness, capture status, buffer sizes |
| `WS` | `/stream` | Real-time stream of classified packets as newline-delimited JSON |

### Example: POST /predict

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Average Packet Size": 80.0,
    "Subflow Fwd Bytes": 40.0,
    "Destination Port": 80.0,
    "Init_Win_bytes_forward": 0.0,
    "PSH Flag Count": 0.0,
    "FIN Flag Count": 0.0,
    "ACK Flag Count": 1.0,
    "Flow Bytes/s": 1500000.0,
    "Flow Packets/s": 2000.0,
    "Fwd Packets/s": 2000.0,
    "Total Fwd Packets": 2000.0
  }'
```

**Response:**

```json
{
  "stage": "cloud",
  "final_label": "DDoS",
  "severity": "high",
  "edge": {
    "is_suspicious": true,
    "is_attack": true,
    "confidence": 1.000,
    "prediction": "Attack",
    "reason": "attack_confident",
    "edge_latency_ms": 0.21
  },
  "cloud": {
    "attack_type": "DDoS",
    "confidence": 1.000,
    "top_3_classes": [
      {"class": "DDoS",           "prob": 1.0},
      {"class": "Normal Traffic", "prob": 0.0},
      {"class": "DoS",            "prob": 0.0}
    ],
    "cloud_latency_ms": 0.15
  },
  "pipeline_latency_ms": 0.41
}
```

### Severity mapping

| Severity | Rule |
|---|---|
| `none` | Cloud predicted Normal Traffic (edge false positive) |
| `low` | Confirmed attack below medium thresholds |
| `medium` | Brute Force / Web Attacks / Port Scanning with conf ≥ 0.55, or any attack conf ≥ 0.65 |
| `high` | DDoS / DoS / Bots with conf ≥ 0.65, or any attack with conf ≥ 0.90 |

---

## 8. Phase 6 — Kafka Streaming Pipeline

For production streaming (live traffic via the API → Kafka → hybrid pipeline), a bridge connects the FastAPI WebSocket to Kafka:

```
Live traffic (eth2)
    │
    ▼
FastAPI WebSocket (/stream)  ─── kafka_pipeline/producer.py ──▶  Kafka topic: network-events
                                  [WS client → Kafka bridge]
                                                                          │
                                                                          ▼
                                                              hybrid/edge_detector.py
                                                                [KafkaConsumer]
                                                                          │
                                                          DROP ◀── Normal (conf ≥ 0.95)
                                                                          │
                                                        suspicious-traffic│
                                                                          ▼
                                                              hybrid/cloud_analyzer.py
                                                                [KafkaConsumer]
                                                                          │
                                                         ┌────────────────┤
                                                         ▼                ▼
                                               classified-attacks       alerts
                                               (all forwarded)     (attacks only)
                                                         │
                                                         ▼
                                               hybrid/orchestrator.py
                                               [multi-topic consumer]
                                               → data/hybrid_metrics.json (every 10s)
                                               → http://localhost:9100/metrics (Prometheus)
```

### Orchestrator metrics (hybrid_metrics.json)

```json
{
  "packets_total": 0,
  "packets_forwarded_to_cloud": 0,
  "attacks_confirmed_by_xgboost": 0,
  "edge_false_positive_rate": 0.0,
  "bandwidth_saved_pct": 0.0,
  "avg_edge_latency_ms": 0.0,
  "avg_cloud_latency_ms": 0.0,
  "p95_total_latency_ms": 0.0,
  "attack_type_distribution": {}
}
```

---

## 9. Infrastructure

**File:** `infra/docker-compose.yml`

```bash
cd infra && docker compose up -d
```

| Service | Image | Port | Role |
|---|---|---|---|
| `nids-kafka` | apache/kafka:4.1.0 (KRaft) | 29092 (external) | Message broker — no ZooKeeper |
| `nids-kafka-ui` | provectuslabs/kafka-ui | 8081 | Visual Kafka explorer |
| `nids-neo4j` | neo4j:5.26-community | 7474 (browser), 7687 (Bolt) | Attack graph storage |

Kafka topics are auto-created on first publish:

| Topic | Partitions | Consumers |
|---|---|---|
| `network-events` | 3 | Edge Detector |
| `suspicious-traffic` | 3 | Cloud Analyzer |
| `classified-attacks` | 3 | Orchestrator |
| `alerts` | 3 | Orchestrator, Dashboard, Neo4j |
| `model-weights` | 3 | FL Server |
| `global-model` | 3 | FL Clients |

---

## 10. Model Performance Summary

### Centralized models — test set (504,151 samples)

| Model | Accuracy | F1 macro | Size | Notes |
|---|---|---|---|---|
| RandomForest (baseline) | 99.43% | 0.863 | 21.6 MB | 200 trees, depth 20 |
| XGBoost (Optuna) | 99.44% | 0.862 | 9.6 MB | **Champion multiclass** |
| MLP binary v1 | 99.20% | 0.986 | 1.0 MB | Standard CE |
| MLP binary v2 | 99.25% | 0.987 | 1.0 MB | Focal loss |
| MLP binary v3 | 99.38% | 0.989 | 1.0 MB | **Champion binary / FL start** |

### Hybrid architecture — test set comparison

| | XGBoost alone | MLP→XGBoost | Savings |
|---|---|---|---|
| Accuracy | 99.44% | 99.44% | −0.006 pp |
| F1 macro | 0.8620 | 0.8575 | −0.005 |
| XGBoost calls / 1000 pkts | 1,000 | 207 | **−79.3%** |
| Cloud bandwidth | 100% | 20.7% | **−79.3%** |
| Edge missed attacks | — | 0.07% | — |

---

## 11. Project File Structure

```
nids-fl-project/
│
├── data/
│   ├── cicids2017_cleaned.csv          # Raw dataset (685 MB)
│   ├── processed/                      # Preprocessed npz splits
│   │   ├── train.npz / val.npz / test.npz
│   │   └── fl_client_{1,2,3}.npz
│   └── hybrid_metrics.json             # Live orchestrator metrics
│
├── models/
│   ├── mlp_binary_ckpt.keras           # Edge MLP (64→32→16, EarlyStopping ckpt)
│   ├── mlp_binary.keras                # Full trained binary MLP
│   └── final/
│       ├── best_centralized_multiclass.json  # XGBoost cloud model
│       ├── best_mlp_binary.keras             # Best centralized binary MLP
│       ├── baseline_rf.joblib                # RandomForest baseline
│       ├── scaler.joblib                     # Fitted StandardScaler
│       ├── label_encoder.joblib              # 7-class LabelEncoder
│       ├── selected_features.json            # 11 feature names
│       └── MANIFEST.json                     # All model metrics & metadata
│
├── notebooks/
│   ├── 01_explore_dataset.ipynb        # EDA + class distribution
│   ├── 02_random_forest.ipynb          # RF baseline + CV
│   ├── 03_xgboost_optuna.ipynb         # XGBoost + Optuna HPO
│   ├── 04_mlp_centralized.ipynb        # MLP binary v1/v2/v3
│   └── 06_hybrid_vs_centralized.ipynb  # Architecture comparison
│
├── federated_learning/
│   ├── model.py                        # build_mlp(), evaluate_mlp()
│   ├── aggregation.py                  # FedAvg, weight serialization
│   ├── preprocessing.py                # FeatureScaler
│   ├── fl_server.py                    # FedAvg server (Kafka)
│   └── fl_client.py                    # FL client (Kafka)
│
├── hybrid/
│   ├── edge_detector.py                # Stage 1: binary MLP filter
│   ├── cloud_analyzer.py               # Stage 2: XGBoost 7-class
│   └── orchestrator.py                 # Pipeline metrics + Prometheus
│
├── api/
│   ├── main.py                         # FastAPI app (full pipeline)
│   ├── capture.py                      # pyshark async generator
│   └── features.py                     # Feature extraction + FlowTracker
│
├── kafka_pipeline/
│   ├── producer.py                     # WS→Kafka bridge
│   └── test_connection.py
│
├── attacks/
│   ├── attack_simulator.py             # Traffic attack simulator
│   └── demo_sequence.py               # Demo attack sequences
│
├── dashboard/                          # React + Vite frontend
│   └── src/
│       └── App.tsx
│
├── neo4j_graph/
│   └── graph_writer.py                 # Attack graph persistence
│
├── infra/
│   └── docker-compose.yml              # Kafka 4.1 + Neo4j + Kafka-UI
│
├── docs/images/                        # Generated evaluation plots
│   ├── hybrid_latency.png
│   ├── hybrid_traffic_pie.png
│   ├── hybrid_scatter.png
│   ├── hybrid_confusion.png
│   └── hybrid_f1_class.png
│
└── tests/
    ├── test_model.py
    ├── test_features.py
    └── test_kafka.py
```

---

## 12. Running the Full Stack

### Prerequisites

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd infra && docker compose up -d   # Kafka + Neo4j
```

### Option A — API only (no Kafka, direct inference)

```bash
# Start the API — models load automatically on startup
uvicorn api.main:app --host 0.0.0.0 --port 8001

# Test classification
curl http://localhost:8001/health
curl http://localhost:8001/pipeline/status
curl http://localhost:8001/stats

# Classify a packet manually
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"Average Packet Size": 80, "Subflow Fwd Bytes": 40, ...'

# Stream live results
websocat ws://localhost:8001/stream
```

### Option B — Full Kafka pipeline

```bash
# 1. Start infrastructure
cd infra && docker compose up -d

# 2. Start API (captures live traffic and streams to WS)
uvicorn api.main:app --host 0.0.0.0 --port 8001

# 3. Bridge: live packets → Kafka
python -m kafka_pipeline.producer

# 4. Edge detector (Stage 1)
python -m hybrid.edge_detector --threshold 0.95

# 5. Cloud analyzer (Stage 2)
python -m hybrid.cloud_analyzer

# 6. Orchestrator (metrics + Prometheus)
python -m hybrid.orchestrator

# 7. (Optional) Federated learning
python -m federated_learning.fl_server --rounds 10 --clients 3 &
python -m federated_learning.fl_client --id 1 &
python -m federated_learning.fl_client --id 2 &
python -m federated_learning.fl_client --id 3
```

### Monitoring

| URL | Description |
|---|---|
| `http://localhost:8001/docs` | FastAPI interactive Swagger UI |
| `http://localhost:8001/stats` | Live packet + attack statistics |
| `http://localhost:8001/alerts` | Active attack alerts |
| `http://localhost:8081` | Kafka UI (topic browser) |
| `http://localhost:7474` | Neo4j Browser (attack graph) |
| `http://localhost:9100/metrics` | Prometheus metrics (orchestrator) |

---

*Generated: 2026-05-11 | Dataset: CICIDS2017 | Models: XGBoost Optuna + MLP Keras | Stack: FastAPI · Kafka 4.1 · Neo4j · Docker*
