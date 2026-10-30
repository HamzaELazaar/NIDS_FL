# Résumé de session — Pipeline NIDS-FL

## Objectif général

Construire le pipeline de collecte et de transport des données réseau en temps réel pour un
système NIDS basé sur le Federated Learning. Le pipeline doit capturer le trafic réseau,
extraire des features ML-compatibles, les exposer via une API, et les publier dans Kafka.

---

## Environnement technique

| Élément | Détail |
|---|---|
| OS | WSL2 Ubuntu (Linux 6.6 / Windows) |
| Python | 3.12 (venv) |
| Broker | Apache Kafka 4.1.0 KRaft (Docker Desktop) |
| API | FastAPI 0.115 + uvicorn 0.32 |
| Capture | pyshark 0.6 / tshark (wireshark-common) |
| Interface réseau | eth2 (port mirroring du réseau Windows 192.168.1.x) |

---

## 1. Test de connexion Kafka — `kafka_pipeline/test_connection.py`

### Ce qui a été fait
Création d'un script de smoke-test complet pour valider la connectivité Kafka avant de
construire le reste du pipeline.

### Fonctions implémentées

**`test_admin_connection()`**
- Utilise `KafkaAdminClient` de `kafka-python-ng`
- Liste tous les topics existants via `admin.list_topics()`
- Vérifie la présence des 4 topics requis : `network-events`, `model-weights`,
  `global-model`, `alerts`
- Affichage ✅/❌ par topic

**`test_produce(n=5)`**
- Crée un `KafkaProducer` avec `value_serializer` JSON
- Envoie 5 messages de test sur `network-events`
- Confirme avec `flush()`

**`test_consume(expected=5)`**
- Crée un `KafkaConsumer` (`group_id='test-group'`, `auto_offset_reset='earliest'`)
- Lit les messages avec `consumer_timeout_ms=5000`
- Affiche partition + offset + contenu de chaque message

**`main()`**
- Exécute les 3 tests en séquence
- Hints d'erreur : commande `docker compose up` si broker injoignable,
  commande `kafka-topics.sh --create` si topics manquants

### Problèmes rencontrés et solutions

**Problème 1 : `NodeNotReadyError`**
- **Cause** : `KAFKA_ADVERTISED_LISTENERS: 'PLAINTEXT://kafka:9092'` — le broker annonce
  son adresse interne Docker (`kafka`), que Python dans WSL2 ne peut pas résoudre.
- **Solution** : Ajout d'un listener `EXTERNAL` dédié aux clients hôte dans `docker-compose.yml` :
  ```yaml
  KAFKA_LISTENERS: 'PLAINTEXT://0.0.0.0:9092,EXTERNAL://0.0.0.0:29092,CONTROLLER://0.0.0.0:9093'
  KAFKA_ADVERTISED_LISTENERS: 'PLAINTEXT://kafka:9092,EXTERNAL://localhost:29092'
  KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: 'CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT,EXTERNAL:PLAINTEXT'
  ```
- **Résultat** : Python se connecte sur `localhost:29092`, les containers internes
  (Kafka UI) continuent d'utiliser `kafka:9092`.

**Problème 2 : `kafka-topics.sh: not found`**
- **Cause** : L'image `apache/kafka:4.1.0` place les scripts dans `/opt/kafka/bin/`,
  pas dans le PATH.
- **Solution** : Utiliser le chemin complet :
  ```bash
  docker exec nids-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create ...
  ```

**Résultat final**
```
✅ admin    ✅ produce    ✅ consume    🎉 All tests passed
```

---

## 2. Capture réseau — `api/capture.py`

### Ce qui a été fait
Wrapper async autour de pyshark pour exposer la capture live comme un générateur Python
utilisable par FastAPI sans bloquer l'event loop.

### Fonctions implémentées

**`live_capture(interface, bpf_filter)`** — async generator
- Interface par défaut : `eth2`
- Filtre BPF par défaut : `"ip and (tcp or udp or icmp)"`
- Vérifie la disponibilité de l'interface via `list_interfaces()` avant de démarrer
- Logs : `"Capture started on {interface}..."`
- Gestion des erreurs : `TSharkCrashException`, `PermissionError`, `OSError`

**`list_interfaces()`**
- Lance `subprocess.run(["tshark", "-D"])` avec timeout 5s
- Parse la sortie avec regex `r"^(\d+)\.\s+(\S+)"`
- Retourne `list[dict]` avec clés `index` et `name`
- Retourne `[]` proprement si tshark absent

**`capture_to_buffer(buffer, interface, max_size)`**
- Tourne indéfiniment (jusqu'à `CancelledError`)
- Chaque paquet → `extract_features()` → append au `deque`
- Fallback sur `{"raw": str(packet)}` si `features.py` non disponible

### Problèmes rencontrés et solutions

**Problème 1 : Permissions tshark**
- **Cause** : tshark nécessite les droits root pour capturer.
- **Solution** : `setcap` sur dumpcap (pas de sudo) :
  ```bash
  sudo setcap cap_net_raw,cap_net_admin=eip $(which dumpcap)
  ```

**Problème 2 : `RuntimeError: This event loop is already running`**
- **Cause** : `sniff_continuously()` dans `run_in_executor` appelle
  `loop.run_until_complete()` depuis un thread alors que la boucle principale tourne.
- **Solution** : Remplacement par l'API async native de pyshark :
  `packets_from_tshark(callback)` + bridge `asyncio.Queue` :
  ```python
  queue = asyncio.Queue()
  def _enqueue(packet): queue.put_nowait(packet)
  task = asyncio.ensure_future(_capture_task())  # appelle packets_from_tshark(_enqueue)
  while True:
      item = await queue.get()
      if item is _sentinel: break
      yield item
  ```

**Problème 3 : `TypeError: packets_from_tshark() missing 1 required positional argument`**
- **Cause** : `packets_from_tshark()` n'est pas un générateur async mais un coroutine
  qui prend un callback.
- **Solution** : Bridge via `asyncio.Queue` (voir ci-dessus).

**Problème 4 : GC `RuntimeError: Event loop is closed`**
- **Cause** : Après `break` d'un `async for`, Python ne `await` pas automatiquement
  `aclose()` — le GC tente de nettoyer le transport subprocess après fermeture de la boucle.
- **Solution** : Appel explicite `await gen.aclose()` dans un `finally` + `gc.collect()`
  avant `asyncio.sleep(0.2)`.

**Problème 5 : `NotImplementedError` sous uvicorn (Python 3.12)**
- **Cause** : pyshark 0.6 appelle `asyncio.set_child_watcher(asyncio.SafeChildWatcher())`
  dans `_setup_eventloop()` — `SafeChildWatcher` a été supprimé en Python 3.12.
- **Solution** : Monkey-patch à l'import :
  ```python
  from pyshark.capture.capture import Capture as _PysharkCapture
  _PysharkCapture._setup_eventloop = lambda self: None
  ```

---

## 3. Extraction de features — `api/features.py`

### Ce qui a été fait
Extraction des 18 features réseau alignées avec CICIDS2017 pour cohérence
entraînement/inférence.

### Features extraites

| Feature | Type | Source pyshark |
|---|---|---|
| `timestamp` | float | `packet.sniff_timestamp` |
| `protocol` | str | layer détecté (TCP/UDP/ICMP/OTHER) |
| `src_ip` | str | `ip.src` |
| `dst_ip` | str | `ip.dst` |
| `src_port` | int | `tcp.srcport` / `udp.srcport` |
| `dst_port` | int | `tcp.dstport` / `udp.dstport` |
| `packet_length` | int | `ip.len` |
| `ttl` | int | `ip.ttl` |
| `syn_flag` | int (0/1) | `tcp.flags` bit 1 |
| `fin_flag` | int (0/1) | `tcp.flags` bit 0 |
| `rst_flag` | int (0/1) | `tcp.flags` bit 2 |
| `psh_flag` | int (0/1) | `tcp.flags` bit 3 |
| `ack_flag` | int (0/1) | `tcp.flags` bit 4 |
| `urg_flag` | int (0/1) | `tcp.flags` bit 5 |
| `window_size` | int | `tcp.window_size_value` |
| `icmp_type` | int | `icmp.type` |
| `icmp_code` | int | `icmp.code` |
| `payload_length` | int | `ip_len − ip_hdr − transport_hdr` |

### Fonctions implémentées

**`extract_features(packet) → dict | None`**
- Retourne `None` si pas de couche IP (`not hasattr(packet, "ip")`)
- `try/except Exception` global : jamais de crash sur paquet malformé
- Tous les champs castés en `int`/`float`/`str` via helpers `_int()`, `_str()`, `_tcp_flag()`
- Calcul du `payload_length` : TCP = `ip_len − ip_hdr − tcp_hdr`,
  UDP = `udp.length − 8`, ICMP = `ip_len − ip_hdr − 8`

**`normalize_features(features) → dict`**
- Itère sur `FEATURE_KEYS` dans l'ordre canonique fixe
- Valeurs manquantes → 0 (int) ou "" (str) depuis `_DEFAULTS`
- Cast strict selon le type du défaut

**`FEATURE_KEYS`** — liste ordonnée des 18 clés : garantit que le vecteur live
a la même structure que les lignes CICIDS2017.

### Validation
```
✅ All 18 keys present
[1] ICMP  172.31.8.68 → 8.8.8.8  len=84  payload=56  icmp_type=8  icmp_code=0
[2] TCP   172.31.8.68 → 20.189.173.14  len=60  payload=0  flags=[SYN]  win=64240
```

---

## 4. API FastAPI — `api/main.py`

### Ce qui a été fait
Exposition du flux de capture en temps réel via WebSocket et REST.

### Architecture interne

```
lifespan startup → asyncio.create_task(capture_loop())
                          │
                 live_capture(eth2)
                          │
               extract_features() + normalize_features()
                    │              │
             BUFFER deque(1000)   _broadcast() → WS_CLIENTS
```

### Endpoints

| Endpoint | Méthode | Description |
|---|---|---|
| `/stream` | WebSocket | Broadcast temps réel à tous les clients connectés |
| `/packets` | GET | Query buffer : `limit`, `protocol`, `src_ip`, `dst_ip`, `dst_port`, `min_ttl` |
| `/stats` | GET | `total_packets`, `packets_per_second`, `protocols`, `top_5_source_ips`, `top_5_destination_ports` |
| `/health` | GET | `status`, `capture_active`, `uptime_seconds`, `buffer_size`, `ws_clients` |
| `/docs` | GET | Swagger UI auto-généré |

### Détails d'implémentation

**Buffer circulaire** : `deque(maxlen=1000)` — fenêtre glissante des 1000 derniers paquets.

**WebSocket** : `capture_loop()` appelle `_broadcast(json.dumps(normalized))` pour chaque
paquet. `_broadcast()` retire automatiquement les clients morts (try/except sur `send_text`).

**CORS** : `allow_origins=["*"]` — nécessaire pour le dashboard Streamlit sur port différent.

**Lifespan** : `asyncio.create_task(capture_loop())` au startup, `task.cancel()` au shutdown.

### Validation live

```
packets: raw=160  none=0  buffered=160
```
→ 100% des paquets passent `extract_features()`, aucune perte.

---

## 5. Bridge WebSocket → Kafka — `kafka_pipeline/producer.py`

### Ce qui a été fait
Service qui consomme le WebSocket FastAPI et publie chaque paquet dans Kafka.

### Fonctions implémentées

**`stream_to_kafka(producer)`** — async, boucle infinie
- `websockets.connect(WS_URL)` avec `ping_interval=20`, `ping_timeout=10`
- Parse JSON de chaque message WS
- `producer.send(TOPIC, key=src_ip, value=packet)`
- `future.get(timeout=5)` pour surface les erreurs sans bloquer longtemps
- Log : `📨 TCP  192.168.1.8:443 → 20.189.173.14:443`
- Reconnexion auto : backoff exponentiel 1s → 2s → 4s → … → 60s max

**`_report_stats(interval=10.0)`** — async, tâche background
- Toutes les 10 secondes :
  ```
  📊 Stats — sent: 99  msg/s: 5.44  errors: 0
  ```

**`main()`**
- Lit `KAFKA_BROKER`, `TOPIC_NETWORK_EVENTS`, `API_HOST`, `API_PORT` depuis `.env`
- 5 tentatives de connexion Kafka au démarrage (backoff 2^n, max 30s)
- `asyncio.gather(stream_to_kafka(), _report_stats())`
- `producer.flush() + producer.close()` garanti dans le `finally`

### Partitioning

`key=src_ip` → tous les paquets d'une même IP source vont dans la même partition Kafka.
Cela garantit qu'un consumer FL voit l'historique complet d'un hôte (important pour les
features temporelles comme les séquences de flags TCP).

### Validation finale (Kafka UI)

```
Topic: network-events
Message Count: 3291
Partition 0: 1357 messages
Partition 1:   94 messages
Partition 2: 1840 messages
```

Le déséquilibre entre partitions reflète la distribution réelle du trafic
(192.168.1.8 est la source principale → toujours sur la même partition).

---

## 6. Infrastructure Kafka — `infra/docker-compose.yml`

### Modifications apportées

**Dual-listener (fix WSL2 → Docker)**

Problème : WSL2 et les containers Docker sont sur des réseaux virtuels séparés.
Le listener `PLAINTEXT://kafka:9092` n'est résolvable que depuis l'intérieur de Docker.

Solution :
```yaml
ports:
  - "9092:9092"      # interne containers
  - "29092:29092"    # externe WSL2/host
  - "9093:9093"      # controller KRaft

KAFKA_LISTENERS: 'PLAINTEXT://0.0.0.0:9092,EXTERNAL://0.0.0.0:29092,CONTROLLER://0.0.0.0:9093'
KAFKA_ADVERTISED_LISTENERS: 'PLAINTEXT://kafka:9092,EXTERNAL://localhost:29092'
KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: 'CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT,EXTERNAL:PLAINTEXT'
```

**Kafka UI port 8081**

Port 8080 occupé par PostgreSQL/EDB sur la machine hôte.
Kafka UI exposé sur `8081:8080` → `http://localhost:8081`.

**`.env`** mis à jour :
```
KAFKA_BROKER=localhost:29092
CAPTURE_INTERFACE=eth2
```

---

## 7. Résumé des problèmes techniques résolus

| # | Problème | Cause | Solution |
|---|---|---|---|
| 1 | `NodeNotReadyError` Kafka | Listener advertised sur hostname Docker | Dual-listener EXTERNAL:29092 |
| 2 | `kafka-topics.sh not found` | PATH Docker image apache/kafka:4.1.0 | `/opt/kafka/bin/kafka-topics.sh` |
| 3 | Permissions tshark | Capture réseau = root par défaut | `setcap cap_net_raw,cap_net_admin=eip` sur dumpcap |
| 4 | `Event loop already running` | `sniff_continuously` dans `run_in_executor` | API callback `packets_from_tshark` + bridge `asyncio.Queue` |
| 5 | `TypeError: missing argument` | `packets_from_tshark` n'est pas un générateur | Bridge Queue (callback → yield) |
| 6 | `NotImplementedError` Python 3.12 | `SafeChildWatcher` supprimé en 3.12 | Monkey-patch `_setup_eventloop = lambda self: None` |
| 7 | GC `Event loop is closed` | `aclose()` non awaité après `break` | `await gen.aclose()` dans `finally` + `gc.collect()` |
| 8 | `buffer_size: 0` sous uvicorn | `capture_active=true` mais crash silencieux | `exc_info=True` dans le log → révèle `NotImplementedError` |
| 9 | Port 8080 occupé (PostgreSQL) | Kafka UI ne pouvait pas binder | Port remappé sur 8081 |
| 10 | 0 paquets sur eth0 depuis WSL2 | Trafic Windows ne passe pas par eth0 WSL2 | Port mirroring → eth2 |

---

## 8. Flux de données validé bout en bout

```
Réseau physique Windows (192.168.1.8)
    │
    │  [Port mirroring]
    ▼
eth2 (WSL2)
    │
    │  [pyshark LiveCapture — BPF: ip and (tcp or udp or icmp)]
    ▼
extract_features() — 18 features dict (CICIDS2017-aligned)
    │
    │  [deque(maxlen=1000) + WebSocket broadcast]
    ▼
FastAPI ws://localhost:8000/stream
    │
    │  [websockets.connect — key=src_ip]
    ▼
Kafka topic: network-events (3 partitions)
    │
    ▼
3291 messages confirmés dans Kafka UI (localhost:8081)
```

---

## 9. Commandes de lancement

```bash
# Terminal 1 — API FastAPI
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Bridge WS → Kafka
python kafka_pipeline/producer.py

# Terminal 3 — Consumer Kafka (vérification)
docker exec nids-kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic network-events

# Test de connexion Kafka
python kafka_pipeline/test_connection.py

# Test capture + features
python api/features.py

# Kafka UI
http://localhost:8081

# Swagger API
http://localhost:8000/docs
```

---

## 10. Fichiers créés / modifiés

| Fichier | Statut | Description |
|---|---|---|
| `kafka_pipeline/test_connection.py` | Créé | Smoke-tests admin + produce + consume |
| `kafka_pipeline/producer.py` | Créé | Bridge WebSocket → Kafka |
| `api/capture.py` | Créé | Async generator pyshark |
| `api/features.py` | Créé | Extraction 18 features CICIDS2017 |
| `api/main.py` | Créé | FastAPI WS + REST |
| `infra/docker-compose.yml` | Modifié | Dual-listener + Kafka UI port 8081 |
| `.env` | Modifié | `KAFKA_BROKER=localhost:29092`, `CAPTURE_INTERFACE=eth2` |
| `docs/architecture.md` | Créé | Diagramme + choix techniques |

**Commit** : `ac50426` — `feat: live capture pipeline — tshark → FastAPI → Kafka`
