# API reference

FastAPI service defined in [`api/main.py`](../api/main.py). Base URL `http://localhost:8000`.
Interactive documentation is generated automatically at `/docs` (Swagger UI) and `/redoc`.

CORS is fully open (`allow_origins=["*"]`) — appropriate for a lab, **not** for exposure beyond one.

---

## Inference

### `POST /predict`

Classify a single packet through the full hybrid pipeline, in-process.

The body accepts **either** format and auto-detects which one it received (presence of `src_ip` or
`timestamp` selects the live format):

- **11 CICIDS2017 keys** — passed straight to the models: `Average Packet Size`,
  `Subflow Fwd Bytes`, `Destination Port`, `Init_Win_bytes_forward`, `PSH Flag Count`,
  `FIN Flag Count`, `ACK Flag Count`, `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`,
  `Total Fwd Packets`
- **Live capture keys** — `src_ip`, `dst_ip`, `dst_port`, `fin_flag`, … mapped automatically via
  `to_ml_vector()`

Returns the classification with both the edge and cloud verdicts.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Destination Port": 80, "Total Fwd Packets": 12, "Flow Bytes/s": 45000,
       "Average Packet Size": 512, "PSH Flag Count": 1, "FIN Flag Count": 0,
       "ACK Flag Count": 1, "Init_Win_bytes_forward": 8192, "Subflow Fwd Bytes": 6144,
       "Flow Packets/s": 88, "Fwd Packets/s": 88}'
```

Returns `{"error": "Pipeline not ready — models still loading"}` if called before the models finish
loading.

### `POST /inject`

Publish one packet to the `network-events` topic so the **real pipeline** processes it end to end:

```
/inject -> network-events -> edge_detector -> suspicious-traffic
        -> cloud_analyzer -> classified-attacks / alerts
        -> kafka_monitor_loop -> WebSocket -> dashboard
```

Same body format as `/predict`. Missing fields are defaulted (`timestamp` = now,
`protocol` = `TCP`, empty IPs, ports 0). Messages are keyed by `src_ip` so a given source lands on a
stable partition.

If the broker is unreachable, the endpoint **falls back to in-process inference** and returns the
result directly instead of failing. The packet is also echoed immediately to WebSocket clients so
the live view updates without waiting for the Kafka round trip.

Response: `{"status": "queued", "topic": "network-events"}`.

---

## Reading state

All read endpoints serve from bounded in-memory buffers, so they return a recent window rather than
full history.

### `GET /predictions`

| Parameter | Type | Default | Constraint |
|---|---|---|---|
| `limit` | int | 50 | 1–2000 |
| `label` | string | — | filter on `final_label`, e.g. `DDoS` |
| `severity` | string | — | `low` \| `medium` \| `high` |
| `stage` | string | — | `edge_only` \| `cloud` |

```json
{ "total_in_buffer": 1834, "total_matched": 42, "returned": 42, "predictions": [ ... ] }
```

### `GET /alerts`

Confirmed attacks only.

| Parameter | Type | Default | Constraint |
|---|---|---|---|
| `limit` | int | 50 | 1–500 |
| `severity` | string | — | `low` \| `medium` \| `high` |

```json
{ "total_alerts": 210, "total_matched": 17, "returned": 17, "alerts": [ ... ] }
```

### `GET /packets`

Raw captured packets, before inference.

| Parameter | Type | Default | Constraint |
|---|---|---|---|
| `limit` | int | 100 | 1–1000 |
| `protocol` | string | — | `TCP` \| `UDP` \| `ICMP` (case-insensitive) |
| `src_ip` | string | — | exact match |
| `dst_ip` | string | — | exact match |
| `dst_port` | int | — | exact match |

### `GET /stats`

Protocol distribution and attack-type distribution across the current buffer.

---

## Pipeline and models

### `GET /pipeline/status`

Metadata for the loaded models — which artefacts are in memory and whether inference is ready.

### `GET /models`

Registered model metadata read from [`models/final/MANIFEST.json`](../models/final/MANIFEST.json):
model id, role, file, format, size, and the full metrics block for each.

### `GET /hybrid/metrics`

Contents of `data/hybrid_metrics.json`, produced by `hybrid/orchestrator.py`. Returns an empty
object until the orchestrator has run at least once.

### `GET /fl/history`

Federated learning round history from `data/fl_history.json` — per round: `global_f1`,
`global_accuracy`, `global_loss`, per-client `local_f1` / `local_accuracy` / `n_samples` /
`weights_size_kb`, `aggregation_method`, `timestamp`. Falls back to a synthetic history when the
file is absent, so the dashboard always has something to plot.

---

## Graph

### `GET /graph/attacks`

Attack graph topology — nodes and edges built from recent alerts (up to 20 source nodes), or a
synthetic graph when no alert data exists.

### `POST /neo4j/query`

Execute a raw Cypher query against Neo4j.

```bash
curl -X POST http://localhost:8000/neo4j/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (a:Attacker)-[r]->(v:Victim) RETURN a,r,v LIMIT 25"}'
```

> This executes arbitrary Cypher with the configured credentials and performs no statement
> filtering. It is a lab convenience; do not expose this endpoint on an untrusted network.

---

## Operations

### `GET /health`

```json
{
  "status": "healthy",
  "capture_active": false,
  "pipeline_ready": true,
  "uptime_seconds": 412.7,
  "buffer_size": 1834,
  "predictions_size": 1200,
  "alerts_size": 210,
  "ws_clients": 1
}
```

### `WS /stream`

WebSocket feed of live classified packets, newline-delimited JSON. Messages carry a `type`
discriminator and a `data` payload, for example:

```json
{ "type": "packet", "source": "inject", "data": { "src_ip": "10.0.0.4", "protocol": "TCP", ... } }
```

The server registers the socket and broadcasts as events arrive; clients that disconnect are
removed from the broadcast list. Consumed by `dashboard/src/hooks/useWebSocket.ts` through
`WebSocketContext`.

---

## Endpoint summary

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | Classify a feature vector in-process |
| POST | `/inject` | Push a packet into the Kafka pipeline |
| GET | `/predictions` | Recent classified packets (filterable) |
| GET | `/alerts` | Confirmed attacks (filterable) |
| GET | `/packets` | Raw pre-inference packets (filterable) |
| GET | `/stats` | Protocol + attack distribution |
| GET | `/pipeline/status` | Loaded model metadata |
| GET | `/models` | Model registry from MANIFEST.json |
| GET | `/hybrid/metrics` | Orchestrator metrics |
| GET | `/fl/history` | Federated round history |
| GET | `/graph/attacks` | Attack graph topology |
| POST | `/neo4j/query` | Raw Cypher query |
| GET | `/health` | Liveness, capture and pipeline status |
| WS | `/stream` | Live packet stream |
