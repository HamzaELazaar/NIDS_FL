# Troubleshooting

Failure modes derived from the code's own error paths, the startup scripts' checks, and the
configuration mismatches present in the repository.

---

## Kafka

### `NoBrokersAvailable` / "Kafka not reachable"

`scripts/start_pipeline.sh` checks this before starting anything and exits early.

**Cause, almost always: the wrong port.** `.env.example` ships `KAFKA_BROKER=localhost:9092`, but:

| Where you are | Correct address |
|---|---|
| Host / WSL2 (Python code, scripts) | `localhost:29092` |
| Inside the Docker network (kafka-ui → kafka) | `kafka:9092` |

Fix by setting `KAFKA_BROKER=localhost:29092` in `.env`, or pass `--broker localhost:29092`.

Otherwise verify the stack is up:

```bash
docker compose -f infra/docker-compose.yml ps
docker compose -f infra/docker-compose.yml logs kafka
python kafka_pipeline/test_connection.py
```

The broker has a healthcheck on a 10 s interval with 5 retries — give it up to a minute after
`up -d` before concluding it is broken.

### Messages are produced but no stage reacts

Consumers use `auto_offset_reset="latest"`, so anything published *before* a consumer started is
never seen. Start the consumers first, then generate traffic. If you need to replay history, change
the offset reset explicitly.

### Only one FL client's weights arrive

`KAFKA_NUM_PARTITIONS` is 3 to match 3 clients. If you increase `FL_NUM_CLIENTS` without increasing
partitions, clients contend for partitions and rounds time out. Raise both together.

---

## Models and inference

### `/predict` returns `{"error": "Pipeline not ready — models still loading"}`

The models load during FastAPI's `lifespan` startup. TensorFlow plus a 9.6 MB XGBoost model takes
time on a cold start. Poll `/health` until `"pipeline_ready": true`.

If it never becomes ready, check the API log — the champion artefacts must exist:

```
models/final/best_centralized_multiclass.json
models/final/best_mlp_binary.keras
models/final/scaler.joblib
models/final/label_encoder.joblib
```

### Shape mismatch loading the binary MLP

`build_mlp()` omits BatchNorm on the **last** hidden layer by default. The centralised binary MLP
was trained with BatchNorm on every layer, so warm-starting it requires:

```python
build_mlp(input_dim=11, n_classes=2, hidden_layers=(256, 128, 64, 32), all_batchnorm=True)
```

Omitting `all_batchnorm=True` produces a layer-count/shape mismatch against
`best_mlp_binary_weights_init.pkl`.

### Live accuracy is far worse than notebook accuracy

The suspect is the mapping between the 23 captured features and the 11 CICIDS2017 columns
(`ML_FEATURE_KEYS` in `api/features.py`). Two specifics to check:

- `Flow Packets/s` and `Fwd Packets/s` are filled from the **same** value (`packet_freq_1s`).
- Flow features come from a **1-second rolling window keyed by source IP**, whereas CICIDS2017
  features were computed over complete bidirectional flows. Short or bursty traffic will not produce
  comparable values.

Confirm the scaler is applied — every stage must use `FeatureScaler`, never raw values.

### Web Attacks and Bots are never detected

Expected. F1 is 0.2804 for Web Attacks and 0.7671 for Bots, on classes representing 0.085% and
0.077% of the training data. This is a model limitation recorded in
[`models/final/MANIFEST.json`](../models/final/MANIFEST.json), not a runtime fault. See
[models.md](models.md).

---

## Federated learning

### Rounds time out waiting for clients

The server waits `--timeout` seconds (default 120) for `--clients` weight messages. Check each
client log in `logs/fl/client_*.log`. Note `federated_average()` has `min_clients=2`, so a round can
still complete with one client missing — a *degraded* round is not always an obvious failure.

### `ValueError: Client k, layer i : shape ... ne correspond pas`

A client is running a different architecture from the others. All clients must use the same
`FL_HIDDEN = (256, 128, 64, 32)` and `FL_INPUT_DIM = 11`. This usually means one process is running
stale code — restart all of them together via `./scripts/stop_all.sh && ./scripts/start_all.sh`.

### `Reçu N clients, minimum requis : 2`

Fewer than `min_clients` reported in. Start all three clients, and confirm they can reach the broker.

### Scripts run but nothing starts

Both `start_all.sh` and `start_pipeline.sh` invoke `./venv/bin/python` **literally**. The virtualenv
must be at `./venv` inside the project root; an activated environment elsewhere is not used.

---

## Capture

### No packets captured

1. `tshark` must be installed and callable — `pyshark` shells out to it.
2. The user needs capture privileges: `sudo usermod -aG wireshark "$USER"`, then re-login.
3. `CAPTURE_INTERFACE` must name a real interface (`ip link show`). The default `eth0` is often not
   the mirrored interface — the French architecture notes describe an `eth2` mirror setup.
4. The BPF filter is `ip and (tcp or udp or icmp)`; non-IP traffic is discarded before Python.

If capture cannot be made to work, the whole system is still demonstrable:

```bash
python scripts/traffic_gen.py --direct --attack-type mixed --duration 60
```

`--direct` posts to `/inject` and exercises the complete Kafka path without touching a NIC.

---

## Dashboard

### Charts show data but the backend is down

Several helpers in `dashboard/src/lib/api.ts` return bundled fallbacks (e.g. `FL_HISTORY_FALLBACK`)
when a request fails, so the UI renders plausible-looking data offline. Confirm against the API
directly:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/fl/history
```

### Hybrid page is empty

`GET /hybrid/metrics` serves `data/hybrid_metrics.json`, which is `{}` until the orchestrator has
run:

```bash
python -m hybrid.orchestrator --broker localhost:29092
```

### CORS or connection errors from a non-localhost origin

The API allows all origins, so a CORS error points at the URL rather than the policy. Set
`VITE_API_URL` and `VITE_WS_URL` in `dashboard/.env.local` to the address the browser can actually
reach — `localhost` in a browser on a different machine resolves to that machine, not the API host.

---

## Neo4j

### Connection refused on bolt://localhost:7687

Container not up, or credentials changed in one place only. `NEO4J_PASSWORD` in `.env` must match
`NEO4J_AUTH` in `infra/docker-compose.yml` (both default to `changeme`).

### APOC procedures unavailable

The compose file sets `NEO4J_PLUGINS: '["apoc"]'` and unrestricts `apoc.*`. Plugin download happens
on first start and needs network access — check `docker compose logs neo4j`.

---

## Ports in use

| Port | Service |
|---|---|
| 5173 | Dashboard (Vite) |
| 8000 | FastAPI |
| 8081 | Kafka UI |
| 9092 / 29092 / 9093 | Kafka internal / external / controller |
| 7474 / 7687 | Neo4j browser / bolt |
| 9200 | Orchestrator Prometheus endpoint |

`start_pipeline.sh` and `start_all.sh` `pkill` their own stale processes on startup, so a "port in
use" error usually points at a *different* application, not a previous run.
