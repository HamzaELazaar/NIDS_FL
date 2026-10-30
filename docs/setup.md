# Setup

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | TensorFlow 2.18 / Keras 3.6 constrain the upper bound |
| Docker + Compose | recent | Runs Kafka, Kafka UI, Neo4j |
| Node.js | 18+ | Dashboard uses Vite 8 and React 19 |
| tshark / Wireshark | any | Required by `pyshark` for live capture |
| OS | Linux or WSL2 | Live capture assumes a Linux interface; the rest is portable |

Live packet capture needs elevated privileges or membership of the `wireshark` group:

```bash
sudo apt install tshark
sudo usermod -aG wireshark "$USER"   # log out and back in
```

Without capture you can still run the entire pipeline using `POST /inject` and
`scripts/traffic_gen.py`, which is how the demo path works.

## 1. Infrastructure

```bash
docker compose -f infra/docker-compose.yml up -d
docker compose -f infra/docker-compose.yml ps
```

This starts three containers:

| Container | Image | Ports |
|---|---|---|
| `nids-kafka` | `apache/kafka:4.1.0` | 9092 (internal), 29092 (host), 9093 (controller) |
| `nids-kafka-ui` | `ghcr.io/kafbat/kafka-ui` | 8081 → 8080 |
| `nids-neo4j` | `neo4j:5.26-community` | 7474 (browser), 7687 (bolt) |

Kafka is configured in **KRaft mode** — there is no ZooKeeper container. Topics are auto-created
(`KAFKA_AUTO_CREATE_TOPICS_ENABLE: true`) with 3 partitions by default, matching the 3 FL clients.

Wait for the Kafka healthcheck to pass before starting Python services; the healthcheck polls
`kafka-broker-api-versions.sh` every 10 s.

## 2. Python environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`requirements.txt` pins exact versions. The heavy entries are `tensorflow==2.18.0`,
`xgboost==2.1.1`, `optuna==3.6.0` and `pyshark==0.6`.

Edit `.env` if your broker is not on the default port — see [configuration.md](configuration.md)
for every variable.

> **Broker port gotcha:** `.env.example` ships `KAFKA_BROKER=localhost:9092`, but the shell scripts
> and the docker-compose external listener use **`localhost:29092`**. From the host or WSL2, use
> `29092`. `9092` only works from inside the Docker network.

## 3. Dataset (optional — only for training)

```bash
./scripts/setup_dataset.sh
```

Downloads and prepares the CICIDS2017 cleaned/preprocessed dataset
(`ericanacletoribeiro/cicids2017-cleaned-and-preprocessed`, 2,520,751 rows). Requires Kaggle
credentials. Pre-trained champion models are already committed under `models/final/`, so this step
is unnecessary if you only want to run inference.

## 4. Run the inference pipeline

```bash
./scripts/start_pipeline.sh
```

Starts three processes and writes logs to `logs/pipeline/`:

1. **FastAPI** (`uvicorn api.main:app`) on port 8000
2. **Edge detector** — `--threshold 0.95 --batch-size 32`
3. **Cloud analyzer**

The script verifies the broker is reachable before starting anything, kills stale processes from a
previous run, waits for `/health` to answer before launching the consumers, and records PIDs in
`logs/pipeline/pids`.

Options:

```bash
./scripts/start_pipeline.sh --threshold 0.90 --batch-size 64 --broker localhost:29092 --port 8000
```

Stop everything with `./scripts/stop_pipeline.sh`.

## 5. Generate traffic

```bash
python scripts/traffic_gen.py --direct --attack-type mixed --duration 60
```

`--direct` posts to `/inject`, bypassing packet capture entirely. This is the reliable way to
demonstrate the system without privileged capture or a mirrored interface.

## 6. Dashboard

```bash
cd dashboard
npm install
npm run dev          # http://localhost:5173
```

The dashboard defaults to `http://localhost:8000`. To point it elsewhere, create
`dashboard/.env.local`:

```
VITE_API_URL=http://192.168.1.8:8000
VITE_WS_URL=ws://192.168.1.8:8000/stream
```

## 7. Federated training (separate workflow)

FL training is independent of the inference pipeline:

```bash
./scripts/start_all.sh                 # 1 server + 3 clients, 10 rounds, 5 local epochs
./scripts/start_all.sh --daily         # repeat every 24 h
./scripts/start_all.sh --weekly        # repeat every 168 h
./scripts/start_all.sh --hours 48      # custom interval
./scripts/start_all.sh --rounds 20 --epochs 10
```

Logs land in `logs/fl/server.log` and `logs/fl/client_{1,2,3}.log`. Stop with
`./scripts/stop_all.sh`.

> The script invokes `./venv/bin/python` directly, so the virtualenv must live at `./venv` —
> activating a differently-located environment is not enough.

## 8. Optional — orchestrator metrics

```bash
python -m hybrid.orchestrator --broker localhost:29092 --interval 10 --metrics-port 9200
```

Prints an ASCII dashboard, writes `data/hybrid_metrics.json`, and exposes Prometheus metrics at
`http://localhost:9200/metrics`. Until this has run, `data/hybrid_metrics.json` contains `{}` and
the dashboard's hybrid page shows empty state.

## Verifying the install

```bash
curl http://localhost:8000/health          # capture + pipeline status
curl http://localhost:8000/pipeline/status # loaded model metadata
python kafka_pipeline/test_connection.py   # broker reachability
pytest tests/ -v                           # features, Kafka, model
```

A healthy `/health` response reports `"pipeline_ready": true`. If it reports `false`, the models are
still loading or failed to load — check the API log.

## Service URLs

| Service | URL |
|---|---|
| Dashboard | http://localhost:5173 |
| API (Swagger UI) | http://localhost:8000/docs |
| API (ReDoc) | http://localhost:8000/redoc |
| Kafka UI | http://localhost:8081 |
| Neo4j Browser | http://localhost:7474 (`neo4j` / `changeme`) |
| Prometheus metrics | http://localhost:9200/metrics |

Common failures and their fixes: [troubleshooting.md](troubleshooting.md).
