# Configuration

## Environment variables

Copy [`.env.example`](../.env.example) to `.env`. Every variable and its shipped default:

### Kafka

| Variable | Default | Notes |
|---|---|---|
| `KAFKA_BROKER` | `localhost:9092` | **See the port warning below** |
| `TOPIC_NETWORK_EVENTS` | `network-events` | Capture → edge detector |
| `TOPIC_MODEL_WEIGHTS` | `model-weights` | FL clients → FL server |
| `TOPIC_GLOBAL_MODEL` | `global-model` | FL server → FL clients |
| `TOPIC_ALERTS` | `alerts` | Confirmed attacks |

> **Port warning.** `.env.example` ships `localhost:9092`, but docker-compose maps the *external*
> listener to **`29092`** and every shell script defaults to `localhost:29092`. From WSL2 or the
> host, use `29092`; `9092` is the inter-container listener and only resolves inside the Docker
> network. A broker that "isn't reachable" is almost always this.

Two further topics are used by the pipeline but are not in `.env.example`, being hardcoded in
`hybrid/` and `federated_learning/`: `suspicious-traffic`, `classified-attacks`, and `fl-rounds`.

### Neo4j

| Variable | Default |
|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` |
| `NEO4J_USER` | `neo4j` |
| `NEO4J_PASSWORD` | `changeme` |

The same credentials are baked into `infra/docker-compose.yml` (`NEO4J_AUTH`). Change both together.
These are lab defaults committed to the repository — replace them anywhere they could be reached.

### API

| Variable | Default |
|---|---|
| `API_HOST` | `0.0.0.0` |
| `API_PORT` | `8000` |

### Capture

| Variable | Default | Notes |
|---|---|---|
| `CAPTURE_INTERFACE` | `eth0` | The WSL2 interface receiving mirrored traffic |

### Federated learning

| Variable | Default |
|---|---|
| `FL_NUM_CLIENTS` | `3` |
| `FL_NUM_ROUNDS` | `10` |
| `FL_LOCAL_EPOCHS` | `5` |

Kafka's `KAFKA_NUM_PARTITIONS` is set to 3 in docker-compose specifically to match
`FL_NUM_CLIENTS`. If you raise the client count, raise the partition count with it.

---

## Dashboard variables

Set in `dashboard/.env.local` (Vite only exposes variables prefixed `VITE_`):

| Variable | Default |
|---|---|
| `VITE_API_URL` | `http://localhost:8000` |
| `VITE_WS_URL` | `ws://localhost:8000/stream` |

---

## CLI flags

### `hybrid/edge_detector.py`

| Flag | Default | Effect |
|---|---|---|
| `--threshold` | `0.95` | Confidence gate. Lower ⇒ more packets dropped at the edge, higher risk of a missed attack. Higher ⇒ more escalated, more stage-2 cost. |
| `--batch-size` | `32` | Packets per Keras `predict` call. Larger ⇒ better throughput, higher per-packet latency. |
| `--broker` | `localhost:29092` | Kafka bootstrap server |

### `hybrid/cloud_analyzer.py`

| Flag | Default | Effect |
|---|---|---|
| `--broker` | `localhost:29092` | Kafka bootstrap server |
| `--log-interval` | `50` | Log every N processed packets |

### `hybrid/orchestrator.py`

| Flag | Default | Effect |
|---|---|---|
| `--broker` | `localhost:29092` | Kafka bootstrap server |
| `--interval` | `10` | Seconds between metric saves / dashboard prints |
| `--metrics-port` | `9200` | Prometheus endpoint port |
| `--no-prometheus` | off | Disable the HTTP metrics server |

### `federated_learning/fl_server.py`

| Flag | Default | Effect |
|---|---|---|
| `--rounds` | `10` | Number of FedAvg rounds |
| `--clients` | `3` | Clients expected per round |
| `--broker` | `localhost:29092` | Kafka bootstrap server |
| `--timeout` | `120` | Seconds to wait for all clients before proceeding |
| `--repeat-hours` | `0` | 0 = one-shot; >0 = repeat on that interval |

### `federated_learning/fl_client.py`

| Flag | Default | Effect |
|---|---|---|
| `--id` | — | Client identity (1, 2, 3) — selects the local partition |
| `--rounds` | `10` | Rounds to participate in |
| `--epochs` | `5` | Local training epochs per round |
| `--broker` | `localhost:29092` | Kafka bootstrap server |
| `--repeat-hours` | `0` | Continuous mode |

### `scripts/start_pipeline.sh`

`--threshold`, `--batch-size`, `--broker`, `--port`

### `scripts/start_all.sh`

`--daily` (24 h), `--weekly` (168 h), `--hours N`, `--rounds N`, `--epochs N`

### `scripts/traffic_gen.py`

`--direct` (POST to `/inject`, bypassing capture), `--attack-type` (e.g. `mixed`), `--duration`
(seconds)

---

## Tuning notes

**Edge threshold** is the main dial. At `0.95` the system drops only very confident benign packets.
Lowering it toward `0.90` shifts work away from XGBoost and raises throughput at the cost of recall;
raising it pushes almost everything to stage 2 and the hybrid design stops paying for itself.

**Batch size** trades latency for throughput. With a batch of 32 a packet waits for up to 31 others
before being scored. For latency-sensitive demos, lower it; for throughput benchmarks, raise it.

**`min_clients`** in `federated_average()` defaults to 2, so a round can complete with one client
missing. Combined with `--timeout`, this determines whether a slow client stalls training or is
simply skipped for that round.

---

## Hardcoded paths worth knowing

| Path | Written / read by |
|---|---|
| `data/fl_history.json` | FL server; served by `GET /fl/history` |
| `data/hybrid_metrics.json` | Orchestrator (atomic rename); served by `GET /hybrid/metrics` |
| `data/confirmed_attacks.ndjson` | Persisted alerts |
| `models/final/best_mlp_binary_weights_init.pkl` | Warm-start weights for FL server and clients |
| `models/final/MANIFEST.json` | Model registry; served by `GET /models` |
| `logs/pipeline/`, `logs/fl/` | Script logs and PID files |
| `./venv/bin/python` | Hardcoded in `start_all.sh` and `start_pipeline.sh` |
