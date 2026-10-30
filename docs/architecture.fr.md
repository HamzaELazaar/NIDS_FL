# NIDS-FL — Architecture

## Vue d'ensemble

Système de détection d'intrusion réseau (NIDS) basé sur le Federated Learning.
Le trafic réseau est capturé en temps réel, transformé en features ML, distribué via Kafka,
puis analysé par des clients FL décentralisés.

---

## Diagramme de flux

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RÉSEAU PHYSIQUE                                │
│                     (Windows — 192.168.1.8)                             │
└────────────────────────────┬────────────────────────────────────────────┘
                             │  Port mirroring (eth2)
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        WSL2 Ubuntu                                      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  api/capture.py                                                  │   │
│  │  pyshark.LiveCapture(eth2)                                       │   │
│  │  BPF: "ip and (tcp or udp or icmp)"                              │   │
│  │  → async generator de paquets bruts                              │   │
│  └──────────────────────┬───────────────────────────────────────────┘   │
│                         │ pyshark Packet                                │
│  ┌──────────────────────▼───────────────────────────────────────────┐   │
│  │  api/features.py                                                 │   │
│  │  extract_features()  → 18 features (timestamp, protocol,         │   │
│  │                         src/dst IP:port, TTL, TCP flags,          │   │
│  │                         window_size, ICMP type/code, payload)    │   │
│  │  normalize_features() → valeurs int/float/str ML-ready           │   │
│  └──────────────────────┬───────────────────────────────────────────┘   │
│                         │ dict JSON                                     │
│  ┌──────────────────────▼───────────────────────────────────────────┐   │
│  │  api/main.py  (FastAPI + uvicorn)                                │   │
│  │                                                                  │   │
│  │  capture_loop() ──► deque(maxlen=1000)                           │   │
│  │       │                                                          │   │
│  │       ├──► WS /stream    (broadcast temps réel)                  │   │
│  │       ├──► GET /packets  (query + filtres)                       │   │
│  │       ├──► GET /stats    (agrégats protocol/IP/port)             │   │
│  │       └──► GET /health   (liveness + uptime)                    │   │
│  └──────────────────────┬───────────────────────────────────────────┘   │
│                         │ WebSocket ws://localhost:8000/stream           │
│  ┌──────────────────────▼───────────────────────────────────────────┐   │
│  │  kafka_pipeline/producer.py                                      │   │
│  │  websockets.connect() → KafkaProducer                           │   │
│  │  key   = src_ip  (partitioning cohérent par source)             │   │
│  │  topic = network-events                                          │   │
│  │  reconnexion auto (backoff exponentiel 1s → 60s)                │   │
│  └──────────────────────┬───────────────────────────────────────────┘   │
│                         │ localhost:29092 (listener EXTERNAL)           │
└─────────────────────────┼───────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────┐
│                    Docker Desktop                                        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Kafka 4.1 KRaft  (nids-kafka)                                  │    │
│  │                                                                 │    │
│  │  Listeners:                                                     │    │
│  │    PLAINTEXT  kafka:9092    ← interne containers                │    │
│  │    EXTERNAL   localhost:29092 ← clients WSL2/host               │    │
│  │    CONTROLLER kafka:9093    ← KRaft consensus                   │    │
│  │                                                                 │    │
│  │  Topics (3 partitions, RF=1) :                                  │    │
│  │    network-events   ← paquets capturés (3291+ msgs)             │    │
│  │    model-weights    ← gradients FL clients → serveur            │    │
│  │    global-model     ← modèle agrégé serveur → clients           │    │
│  │    alerts           ← anomalies détectées                       │    │
│  └───────────────────────────┬─────────────────────────────────────┘    │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐    │
│  │  Kafka UI  (nids-kafka-ui)  → http://localhost:8081             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Neo4j 5.26  (nids-neo4j)                                       │    │
│  │  bolt://localhost:7687  — graphe des connexions suspectes       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌──────────── À venir ────────────┐
                    │                                 │
                    ▼                                 ▼
          federated_learning/               neo4j_graph/
          FL clients (×3)                  graph_writer.py
          consumer network-events          Noeuds IP + arêtes
          entraînement local               protocole/port
                    │
                    ▼
          FL server (agrégation FedAvg)
          producer global-model
```

---

## Composants

### Capture — `api/capture.py`

| Élément | Valeur |
|---|---|
| Librairie | pyshark 0.6 (wrapper tshark) |
| Interface | eth2 (port mirroring du réseau Windows) |
| Filtre BPF | `ip and (tcp or udp or icmp)` |
| Pattern | async generator via `packets_from_tshark()` + `asyncio.Queue` bridge |
| Permissions | `cap_net_raw,cap_net_admin=eip` sur dumpcap (pas de sudo) |

**Pourquoi pyshark ?** Accès aux champs dissectés de tshark (flags TCP, ICMP type/code)
sans parser le binaire pcap manuellement. L'API callback-based est bridgée en async generator
via une queue pour rester compatible avec uvicorn.

**Pourquoi le bridge Queue ?** pyshark ≤ 0.6 appelle `asyncio.SafeChildWatcher` (supprimé
en Python 3.12). Le monkey-patch `_setup_eventloop = lambda self: None` + le bridge
`packets_from_tshark(callback)` → `asyncio.Queue` → `async for` contourne les deux problèmes.

---

### Extraction — `api/features.py`

18 features alignées avec CICIDS2017 :

| Feature | Source | Notes |
|---|---|---|
| timestamp | `packet.sniff_timestamp` | Unix epoch float |
| protocol | layer détecté | TCP / UDP / ICMP / OTHER |
| src_ip, dst_ip | `ip.src`, `ip.dst` | string dotted-decimal |
| src_port, dst_port | `tcp/udp.srcport/dstport` | 0 pour ICMP |
| packet_length | `ip.len` | bytes totaux IP |
| ttl | `ip.ttl` | Time-To-Live |
| syn/fin/rst/psh/ack/urg_flag | `tcp.flags` bits 1/0/2/3/4/5 | 0 ou 1 |
| window_size | `tcp.window_size_value` | 0 si non-TCP |
| icmp_type, icmp_code | `icmp.type`, `icmp.code` | 0 si non-ICMP |
| payload_length | ip_len − ip_hdr − transport_hdr | bytes applicatifs |

**Cohérence entraînement/inférence** : `normalize_features()` itère sur `FEATURE_KEYS`
dans un ordre canonique fixe — le vecteur de features produit en live a exactement la même
structure que les lignes CICIDS2017 utilisées pour l'entraînement.

---

### API — `api/main.py`

| Endpoint | Description |
|---|---|
| `WS /stream` | Broadcast temps réel à N clients simultanés |
| `GET /packets` | Query buffer avec filtres (protocol, IP, port, TTL) |
| `GET /stats` | Agrégats : protocols, top-5 IPs, top-5 ports, pps |
| `GET /health` | Liveness + `capture_active` + `buffer_size` |

**Buffer circulaire** `deque(maxlen=1000)` : fenêtre glissante des 1000 derniers paquets,
thread-safe pour les lectures REST simultanées.

**CORS `allow_origins=["*"]`** : nécessaire pour le dashboard Streamlit (`dashboard/app.py`)
qui tourne sur un port différent.

---

### Bridge — `kafka_pipeline/producer.py`

```
WebSocket → JSON parse → KafkaProducer.send(key=src_ip, value=features)
```

**Partitioning par `src_ip`** : tous les paquets d'une même source IP vont dans
la même partition → un consumer FL voit l'historique complet d'un hôte,
ce qui améliore la qualité des features temporelles (séquences de flags, etc.).

**Reconnexion exponentielle** : backoff 1s → 2s → 4s → … → 60s max sur coupure WebSocket.
Kafka indisponible au démarrage : 5 tentatives avec backoff 2^n (max 30s) avant abandon.

---

### Kafka — `infra/docker-compose.yml`

**Pourquoi KRaft (sans ZooKeeper) ?** Kafka 4.1 ne supporte plus ZooKeeper.
KRaft réduit la stack de 2 services à 1, simplifie le déploiement dev.

**Dual-listener** : nécessaire car WSL2 et les containers Docker sont sur des réseaux
virtuels séparés. Le listener `EXTERNAL://localhost:29092` est annoncé aux clients hôte ;
`PLAINTEXT://kafka:9092` est réservé au trafic inter-containers (Kafka UI → Kafka).

---

## Flux de données bout-en-bout (validé)

```
eth2 mirror
  → pyshark (tshark subprocess)
  → extract_features() : 18 features dict
  → FastAPI deque(1000) + WS broadcast
  → websockets.connect() consumer
  → KafkaProducer key=src_ip
  → topic network-events (3 partitions)
  → 3291+ messages confirmés dans Kafka UI
```

---

## Stack technique

| Couche | Technologie | Version |
|---|---|---|
| Capture | pyshark / tshark | 0.6 / wireshark |
| API | FastAPI + uvicorn | 0.115 / 0.32 |
| WebSocket client | websockets | 13.1 |
| Message broker | Apache Kafka KRaft | 4.1.0 |
| Kafka client | kafka-python-ng | 2.2.3 |
| Graphe | Neo4j Community | 5.26 |
| ML | TensorFlow / Keras | 2.18 / 3.6 |
| Runtime | Python | 3.12 (WSL2 Ubuntu) |
| Containers | Docker Desktop + Compose | — |
