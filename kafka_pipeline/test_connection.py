"""Kafka connectivity smoke-tests for the NIDS-FL pipeline.

Run:
    python kafka_pipeline/test_connection.py
"""

import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient

load_dotenv()

KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "localhost:9092")
REQUIRED_TOPICS: list[str] = [
    os.getenv("TOPIC_NETWORK_EVENTS", "network-events"),
    os.getenv("TOPIC_MODEL_WEIGHTS", "model-weights"),
    os.getenv("TOPIC_GLOBAL_MODEL", "global-model"),
    os.getenv("TOPIC_ALERTS", "alerts"),
]
_TEST_TOPIC = REQUIRED_TOPICS[0]  # network-events


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def _fail(msg: str) -> None:
    print(f"  ❌ {msg}")


def _header(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  🔍 {title}")
    print(f"{'─' * 50}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Admin connection + topic check
# ──────────────────────────────────────────────────────────────────────────────

def test_admin_connection() -> bool:
    """Verify broker reachability and presence of the four required topics.

    Returns:
        True when all required topics are found, False otherwise.
    """
    _header("Admin connection")
    print(f"  Broker: {KAFKA_BROKER}")

    try:
        admin = KafkaAdminClient(
            bootstrap_servers=KAFKA_BROKER,
            client_id="nids-test-admin",
            request_timeout_ms=5_000,
        )
    except Exception as exc:
        _fail(f"Cannot reach broker → {exc}")
        _hint_broker()
        return False

    existing: list[str] = list(admin.list_topics())
    admin.close()
    _ok(f"Broker reachable — {len(existing)} topic(s) found")

    all_present = True
    for topic in REQUIRED_TOPICS:
        if topic in existing:
            _ok(f"Topic present: {topic}")
        else:
            _fail(f"Topic missing:  {topic}")
            all_present = False

    if not all_present:
        _hint_topics()

    return all_present


# ──────────────────────────────────────────────────────────────────────────────
# 2. Producer test
# ──────────────────────────────────────────────────────────────────────────────

def test_produce(n: int = 5) -> bool:
    """Send *n* test messages to the network-events topic.

    Args:
        n: Number of messages to produce.

    Returns:
        True on success, False on error.
    """
    _header(f"Producer — {n} messages → '{_TEST_TOPIC}'")

    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=5_000,
        )
    except Exception as exc:
        _fail(f"Cannot create producer → {exc}")
        return False

    for i in range(n):
        payload = {
            "test_id": i,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "src_ip": f"192.168.1.{10 + i}",
            "dst_ip": "10.0.0.1",
            "protocol": "TCP",
            "bytes": (i + 1) * 100,
            "label": "benign",
        }
        producer.send(_TEST_TOPIC, value=payload)

    producer.flush()
    producer.close()
    _ok(f"{n} messages sent and flushed successfully")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 3. Consumer test
# ──────────────────────────────────────────────────────────────────────────────

def test_consume(expected: int = 5) -> bool:
    """Consume messages from network-events and print each one.

    Args:
        expected: Number of messages we expect to read back.

    Returns:
        True when at least *expected* messages were received, False otherwise.
    """
    _header(f"Consumer — reading from '{_TEST_TOPIC}'")

    try:
        consumer = KafkaConsumer(
            _TEST_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            group_id="test-group",
            auto_offset_reset="earliest",
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            consumer_timeout_ms=5_000,
        )
    except Exception as exc:
        _fail(f"Cannot create consumer → {exc}")
        return False

    received = 0
    for msg in consumer:
        received += 1
        data = msg.value
        print(
            f"    📨 [{received}] partition={msg.partition} offset={msg.offset} "
            f"| test_id={data.get('test_id')} src={data.get('src_ip')}"
        )

    consumer.close()

    if received >= expected:
        _ok(f"{received} message(s) consumed")
        return True

    _fail(f"Only {received}/{expected} message(s) received (timeout after 5 s)")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Hint helpers
# ──────────────────────────────────────────────────────────────────────────────

def _hint_broker() -> None:
    print(
        "\n  💡 Broker unreachable. Check that Kafka is running:\n"
        "     docker compose -f infra/docker-compose.yml ps\n"
        "     docker compose -f infra/docker-compose.yml up -d kafka\n"
    )


def _hint_topics() -> None:
    missing = ", ".join(REQUIRED_TOPICS)
    print(
        "\n  💡 To create missing topics run:\n"
        "     docker exec nids-kafka kafka-topics.sh \\\n"
        "       --bootstrap-server localhost:9092 --create \\\n"
        f"       --topic <topic-name> --partitions 3 --replication-factor 1\n"
        f"  Required topics: {missing}\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all connectivity tests in sequence and report a final summary."""
    print("\n🚀 NIDS-FL Kafka connectivity tests")
    print(f"   Broker : {KAFKA_BROKER}")
    print(f"   Topics : {', '.join(REQUIRED_TOPICS)}")

    results: dict[str, bool] = {}

    try:
        results["admin"] = test_admin_connection()
        if not results["admin"]:
            sys.exit(1)

        results["produce"] = test_produce()
        results["consume"] = test_consume()

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)

    # Summary
    _header("Summary")
    passed = sum(results.values())
    total = len(results)
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")

    print(f"\n  {'🎉 All' if passed == total else f'⚠️  {passed}/{total}'} tests passed\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
