# Development

## Layout

```
api/                 FastAPI service, capture, feature extraction
attacks/             Attack simulator + scripted demo sequences
dashboard/           React 19 + TS + Vite frontend
data/                Runtime artefacts (fl_history, hybrid_metrics, confirmed_attacks)
docs/                This documentation set + generated figures
federated_learning/  FedAvg server, client, aggregation, model, preprocessing
hybrid/              Edge detector, cloud analyzer, orchestrator
infra/               docker-compose.yml
kafka_pipeline/      Producer + connectivity test
models/              Trained artefacts, metrics, MANIFEST.json
neo4j_graph/         Graph writer
notebooks/           EDA + training notebooks
scripts/             start/stop, traffic generation, benchmarks
tests/               pytest suite
```

Every Python package is importable as a module — services are started with `python -m`
(`python -m hybrid.edge_detector`), not by path. Modules that need the project root on `sys.path`
insert it explicitly at the top of the file.

## Conventions

Established by the existing code, worth matching:

- **Module docstring first.** Every service module opens with a docstring stating its role, the
  topics it consumes and produces, and a `Usage:` block with real command lines. This is the primary
  in-repo documentation and it is reliable — keep it accurate when you change behaviour.
- **`from __future__ import annotations`** at the top of every module.
- **Type hints throughout**, including return types on private helpers.
- **Sectioned files.** Long modules are divided by `# ── Section ──…` rule comments.
- **Colour-coded logging.** Each service defines a `_ColorFormatter` and an emoji-prefixed logger
  (`🟢 [Serveur]`) so multiple tails are distinguishable.
- **Private helpers prefixed `_`**, module constants in `UPPER_SNAKE`.
- **Atomic writes** for any file another process reads — write to a temp path, then `os.replace`.

Formatting tools are pinned in `requirements.txt`: `black==24.10.0`, `isort==5.13.2`.

```bash
black . && isort .
```

> Docstrings and log messages are a mix of English and French across the codebase. New code should
> use English, but existing French strings are load-bearing in places (they appear in `ValueError`
> messages) — do not mass-translate them without checking what reads them.

## Tests

```bash
pytest tests/ -v
pytest tests/test_features.py -v
```

| File | Covers |
|---|---|
| `tests/test_features.py` | Feature extraction, normalisation, ML vector mapping |
| `tests/test_kafka.py` | Producer/consumer round trip |
| `tests/test_model.py` | Model build, weight get/set, FedAvg correctness |

`pytest-asyncio` is available for the async capture and API paths.

## Adding a new pipeline stage

The pipeline is a chain of independent Kafka consumers, so a new stage is additive:

1. Create the module with a docstring naming its input and output topics.
2. Consume with an explicit `group_id` and `auto_offset_reset="latest"` (live behaviour; use
   `earliest` only if you deliberately want replay).
3. Reuse `federated_learning.preprocessing.FeatureScaler` — do not re-implement scaling, or live
   inference will silently diverge from training.
4. Publish enriched records rather than replacing them: downstream stages and the audit trail depend
   on earlier stages' verdicts being preserved.
5. Add the topic to `_ALL_TOPICS` in `hybrid/orchestrator.py` so it is counted.
6. Add a startup line to `scripts/start_pipeline.sh` and a matching `pkill` in the cleanup block.

## Adding an API endpoint

Endpoints live in `api/main.py` in a single section-commented file. Follow the local pattern: a
`summary=` on the decorator, `Query(...)` with `ge`/`le` bounds for numeric parameters, and a
response shaped `{total_*, returned, <payload key>}` for list endpoints, since the dashboard's
`fetchJSON` helper expects that envelope.

If the endpoint feeds the dashboard, add a typed function to `dashboard/src/lib/api.ts` and its
interface to `dashboard/src/types/index.ts`.

## Adding a dashboard page

1. Create `dashboard/src/pages/YourPage.tsx`.
2. Register the route in `dashboard/src/App.tsx` inside the `<Layout />` route group.
3. Add the nav entry in `dashboard/src/components/layout/Sidebar.tsx`.
4. Fetch through `@tanstack/react-query` and the helpers in `lib/api.ts`; live updates come from
   `useWebSocket` / `WebSocketContext` rather than polling.

UI primitives are shadcn/ui components under `dashboard/@/components/ui/`. Charts use
`react-plotly.js`; the attack graph uses `react-force-graph-2d`.

> Several API helpers carry a bundled fallback constant (e.g. `FL_HISTORY_FALLBACK`) and return it
> when the request fails. This keeps demos working offline, but it also means a page can look
> healthy while the backend is down. When debugging "the data never changes", check whether you are
> looking at a fallback.

## Retraining and promoting a model

1. Train in a notebook or a script under `scripts/`.
2. Run `python scripts/save_best_models.py` — it applies the selection rules in
   `models/final/MANIFEST.json` (criterion, preference, tiebreak) and writes the winners plus their
   metrics.
3. Confirm `validation.loads_ok` is `true` for each promoted artefact in the regenerated manifest.
4. If the binary MLP changed, regenerate `best_mlp_binary_weights_init.pkl` — both the FL server and
   clients warm-start from it, and a stale file causes a shape mismatch at load.

## Benchmarking

```bash
python scripts/benchmark_features.py    # feature extraction throughput
python scripts/bench2.py
```

Results are written to `data/benchmark_results.json` and `.txt`.

## Known rough edges

- `scripts/start_all.sh` and `start_pipeline.sh` hardcode `./venv/bin/python`; a virtualenv
  elsewhere is ignored even when activated.
- `.env.example` advertises broker port 9092 while every script uses 29092.
- `POST /neo4j/query` executes arbitrary Cypher with no filtering.
- CORS is `allow_origins=["*"]`.
- `data/hybrid_metrics.json` is `{}` until the orchestrator has run once.
- `neo4j_graph/graph_writer.py` defines no module-level functions or classes — graph writing is
  driven from the API rather than through a reusable interface.
