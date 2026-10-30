"""Packet feature extraction for the NIDS-FL pipeline.

CICIDS2017 alignment
--------------------
The dataset (cicids2017_cleaned.csv) contains 53 flow-level features computed
by CICFlowMeter over complete TCP/UDP sessions, plus an 'Attack Type' label.
Live capture produces one dict per *packet*, not per flow.

Features are split into two groups:

  FEATURE_KEYS (22) — all features extracted from each live packet,
    including 4 pseudo-flow features from the rolling 1-second window.
    Used by the capture pipeline, Kafka, and the REST API.

  ML_FEATURE_KEYS (11) — the subset aligned with CICIDS2017 column names.
    Used for model training and inference to guarantee schema consistency.

CICIDS2017 → live mapping (11 features)
----------------------------------------
  Packet-level (direct):
    'Destination Port'       ← dst_port
    'FIN Flag Count'         ← fin_flag   (0 or 1 per packet)
    'PSH Flag Count'         ← psh_flag
    'ACK Flag Count'         ← ack_flag
    'Init_Win_bytes_forward' ← window_size
    'Subflow Fwd Bytes'      ← payload_length

  Pseudo-flow (rolling 1-second window per src_ip):
    'Flow Packets/s'         ← packet_freq_1s   (packets/sec from src_ip)
    'Flow Bytes/s'           ← bytes_per_sec_1s (bytes/sec from src_ip)
    'Average Packet Size'    ← avg_packet_size_1s (rolling avg, better than single)
    'Total Fwd Packets'      ← packet_count_1s  (count in last 1s)
    'Fwd Packets/s'          ← packet_freq_1s   (same approximation)

Live-only features (absent from CICIDS2017, used for routing/context only):
  timestamp, protocol, src_ip, dst_ip, src_port, ttl,
  syn_flag, rst_flag, urg_flag, icmp_type, icmp_code, packet_length

CICIDS2017-only features (not approximable per packet):
  Flow Duration, Flow IAT stats, Bwd Packet stats, Active/Idle stats…

Feature glossary
----------------
timestamp       : Unix epoch (float) when the packet was captured.
protocol        : Transport protocol string: "TCP", "UDP", "ICMP", or "OTHER".
src_ip          : Source IP address (string, dotted-decimal).
dst_ip          : Destination IP address (string, dotted-decimal).
src_port        : Source port (int); 0 for ICMP or unknown.
dst_port        : Destination port (int) → CICIDS2017 'Destination Port'.
packet_length   : Total IP payload length (int) → CICIDS2017 'Average Packet Size'.
ttl             : IP Time-To-Live (int).
syn_flag        : TCP SYN flag (0 or 1) — not in CICIDS2017.
fin_flag        : TCP FIN flag (0 or 1) → CICIDS2017 'FIN Flag Count'.
rst_flag        : TCP RST flag (0 or 1) — not in CICIDS2017.
psh_flag        : TCP PSH flag (0 or 1) → CICIDS2017 'PSH Flag Count'.
ack_flag        : TCP ACK flag (0 or 1) → CICIDS2017 'ACK Flag Count'.
urg_flag        : TCP URG flag (0 or 1) — not in CICIDS2017.
window_size     : TCP receive window (int) → CICIDS2017 'Init_Win_bytes_forward'.
icmp_type       : ICMP message type (int) — not in CICIDS2017.
icmp_code       : ICMP message code (int) — not in CICIDS2017.
payload_length      : App-layer payload bytes (int) → CICIDS2017 'Subflow Fwd Bytes'.

Pseudo-flow features (rolling 1-second window per src_ip via FlowTracker):
packet_count_1s     : Packet count from src_ip in last 1s → 'Total Fwd Packets'.
bytes_per_sec_1s    : Byte count from src_ip in last 1s → 'Flow Bytes/s'.
avg_packet_size_1s  : Rolling avg packet size → 'Average Packet Size' (more accurate).
packet_freq_1s      : Packets/sec from src_ip → 'Flow Packets/s', 'Fwd Packets/s'.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger(__name__)

# All features extracted per packet (packet-level + pseudo-flow).
FEATURE_KEYS: list[str] = [
    # --- packet-level ---
    "timestamp",
    "protocol",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "packet_length",
    "ttl",
    "syn_flag",
    "fin_flag",
    "rst_flag",
    "psh_flag",
    "ack_flag",
    "urg_flag",
    "window_size",
    "icmp_type",
    "icmp_code",
    "payload_length",
    # --- pseudo-flow (rolling 1-second window per src_ip) ---
    "packet_count_1s",
    "bytes_per_sec_1s",
    "avg_packet_size_1s",
    "packet_freq_1s",
]

# 11 features aligned with CICIDS2017 column names — used for ML.
ML_FEATURE_KEYS: list[str] = [
    "Destination Port",        # ← dst_port
    "FIN Flag Count",          # ← fin_flag
    "PSH Flag Count",          # ← psh_flag
    "ACK Flag Count",          # ← ack_flag
    "Init_Win_bytes_forward",  # ← window_size
    "Subflow Fwd Bytes",       # ← payload_length
    "Total Fwd Packets",       # ← packet_count_1s
    "Flow Bytes/s",            # ← bytes_per_sec_1s
    "Average Packet Size",     # ← avg_packet_size_1s (rolling, more accurate)
    "Flow Packets/s",          # ← packet_freq_1s
    "Fwd Packets/s",           # ← packet_freq_1s (same value, two CICIDS columns)
]

# Maps live feature names → CICIDS2017 column names.
CICIDS2017_MAPPING: dict[str, str] = {
    "dst_port":           "Destination Port",
    "fin_flag":           "FIN Flag Count",
    "psh_flag":           "PSH Flag Count",
    "ack_flag":           "ACK Flag Count",
    "window_size":        "Init_Win_bytes_forward",
    "payload_length":     "Subflow Fwd Bytes",
    "packet_count_1s":    "Total Fwd Packets",
    "bytes_per_sec_1s":   "Flow Bytes/s",
    "avg_packet_size_1s": "Average Packet Size",
    "packet_freq_1s":     "Flow Packets/s",
}

# Well-known port → application-layer protocol name.
# Used only for the `app_protocol` display field — never for ML.
_PORT_TO_APP: dict[int, str] = {
    20:    "FTP-DATA",
    21:    "FTP",
    22:    "SSH",
    23:    "Telnet",
    25:    "SMTP",
    53:    "DNS",
    67:    "DHCP",
    68:    "DHCP",
    80:    "HTTP",
    110:   "POP3",
    123:   "NTP",
    143:   "IMAP",
    161:   "SNMP",
    443:   "HTTPS",
    465:   "SMTPS",
    587:   "SMTP",
    993:   "IMAPS",
    995:   "POP3S",
    3306:  "MySQL",
    5353:  "mDNS",
    5432:  "PostgreSQL",
    6379:  "Redis",
    8000:  "HTTP",
    8080:  "HTTP",
    8443:  "HTTPS",
    27017: "MongoDB",
}

# Default row: all numerics = 0, strings = "".
_DEFAULTS: dict = {
    "timestamp": 0.0,
    "protocol": "",
    "src_ip": "",
    "dst_ip": "",
    "src_port": 0,
    "dst_port": 0,
    "packet_length": 0,
    "ttl": 0,
    "syn_flag": 0,
    "fin_flag": 0,
    "rst_flag": 0,
    "psh_flag": 0,
    "ack_flag": 0,
    "urg_flag": 0,
    "window_size": 0,
    "icmp_type": 0,
    "icmp_code": 0,
    "payload_length": 0,
    "packet_count_1s": 0,
    "bytes_per_sec_1s": 0,
    "avg_packet_size_1s": 0.0,
    "packet_freq_1s": 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# FlowTracker — rolling 1-second window per src_ip
# ─────────────────────────────────────────────────────────────────────────────

class FlowTracker:
    """Thread-safe rolling-window pseudo-flow feature computer.

    Maintains a 1-second sliding window of (timestamp, packet_length) tuples
    per source IP. Approximates CICIDS2017 flow-level statistics without
    requiring full session tracking.

    Args:
        window_seconds: Width of the rolling window in seconds.
    """

    def __init__(self, window_seconds: float = 1.0) -> None:
        self._window = window_seconds
        self._lock = threading.Lock()
        # src_ip → deque of (timestamp, packet_length)
        self._buckets: defaultdict[str, deque] = defaultdict(deque)

    def update(self, src_ip: str, timestamp: float, packet_length: int) -> dict:
        """Record a packet and return pseudo-flow features for *src_ip*.

        Args:
            src_ip: Source IP address used as the flow key.
            timestamp: Packet capture timestamp (Unix epoch).
            packet_length: IP-level packet length in bytes.

        Returns:
            Dict with keys: packet_count_1s, bytes_per_sec_1s,
            avg_packet_size_1s, packet_freq_1s.
        """
        cutoff = timestamp - self._window
        with self._lock:
            dq = self._buckets[src_ip]
            dq.append((timestamp, packet_length))
            while dq and dq[0][0] < cutoff:
                dq.popleft()

            count = len(dq)
            total_bytes = sum(length for _, length in dq)
            avg_size = total_bytes / count if count else 0.0
            freq = count / self._window

        return {
            "packet_count_1s":    count,
            "bytes_per_sec_1s":   total_bytes,
            "avg_packet_size_1s": round(avg_size, 2),
            "packet_freq_1s":     round(freq, 2),
        }

    def cleanup(self, max_keys: int = 50_000) -> None:
        """Evict stale entries if the tracker grows too large.

        Call periodically (e.g. every 60s) to prevent unbounded memory growth
        when capturing from many distinct source IPs.
        """
        now = time.time()
        cutoff = now - self._window
        with self._lock:
            stale = [k for k, dq in self._buckets.items()
                     if not dq or dq[-1][0] < cutoff]
            for k in stale:
                del self._buckets[k]
            if len(self._buckets) > max_keys:
                # Keep most-recently-seen keys
                sorted_keys = sorted(
                    self._buckets, key=lambda k: self._buckets[k][-1][0]
                )
                for k in sorted_keys[:len(self._buckets) - max_keys]:
                    del self._buckets[k]


# Global instance shared across the capture pipeline.
FLOW_TRACKER = FlowTracker(window_seconds=1.0)


def _int(value, default: int = 0) -> int:
    """Safe int cast; returns *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _str(value, default: str = "") -> str:
    """Safe str cast; returns *default* on failure."""
    try:
        return str(value)
    except Exception:
        return default


def _tcp_flag(flags_hex: str, bit: int) -> int:
    """Extract a single TCP flag bit from the hex flags field.

    Args:
        flags_hex: Raw TCP flags value as a hex string (e.g. '0x002').
        bit: Bit position to test (SYN=1, FIN=0, RST=2, PSH=3, ACK=4, URG=5).

    Returns:
        1 if the bit is set, 0 otherwise.
    """
    try:
        return 1 if (int(flags_hex, 16) >> bit) & 1 else 0
    except (TypeError, ValueError):
        return 0


def extract_features(packet) -> Optional[dict]:
    """Extract a flat feature dict from a pyshark Packet.

    Non-IP packets (ARP, etc.) return None so callers can skip them.
    Every field is cast to a plain Python int / float / str — no pyshark
    objects leak into the returned dict.

    Args:
        packet: A ``pyshark.packet.packet.Packet`` instance.

    Returns:
        Dict with exactly the keys in ``FEATURE_KEYS``, or None if the
        packet has no IP layer.
    """
    try:
        if not hasattr(packet, "ip"):
            return None

        ip = packet.ip
        features: dict = dict(_DEFAULTS)

        # ── Timestamp ────────────────────────────────────────────────────────
        try:
            features["timestamp"] = float(packet.sniff_timestamp)
        except (AttributeError, ValueError):
            features["timestamp"] = time.time()

        # ── IP layer ─────────────────────────────────────────────────────────
        features["src_ip"] = _str(ip.src)
        features["dst_ip"] = _str(ip.dst)
        features["ttl"] = _int(ip.ttl)
        features["packet_length"] = _int(ip.len)

        # ── Pseudo-flow (rolling 1-second window per src_ip) ─────────────────
        pseudo = FLOW_TRACKER.update(
            src_ip=features["src_ip"],
            timestamp=features["timestamp"],
            packet_length=features["packet_length"],
        )
        features.update(pseudo)

        # ── Transport layer ───────────────────────────────────────────────────
        if hasattr(packet, "tcp"):
            tcp = packet.tcp
            features["protocol"] = "TCP"
            features["src_port"] = _int(tcp.srcport)
            features["dst_port"] = _int(tcp.dstport)
            features["window_size"] = _int(tcp.window_size_value if hasattr(tcp, "window_size_value") else tcp.window)

            flags = _str(getattr(tcp, "flags", "0x000"))
            features["fin_flag"] = _tcp_flag(flags, 0)
            features["syn_flag"] = _tcp_flag(flags, 1)
            features["rst_flag"] = _tcp_flag(flags, 2)
            features["psh_flag"] = _tcp_flag(flags, 3)
            features["ack_flag"] = _tcp_flag(flags, 4)
            features["urg_flag"] = _tcp_flag(flags, 5)

            # Payload = IP length − IP header − TCP header
            ip_hdr = _int(ip.hdr_len) if hasattr(ip, "hdr_len") else 20
            tcp_hdr = _int(tcp.hdr_len) if hasattr(tcp, "hdr_len") else 20
            features["payload_length"] = max(0, features["packet_length"] - ip_hdr - tcp_hdr)

        elif hasattr(packet, "udp"):
            udp = packet.udp
            features["protocol"] = "UDP"
            features["src_port"] = _int(udp.srcport)
            features["dst_port"] = _int(udp.dstport)
            # UDP length field includes the 8-byte header
            features["payload_length"] = max(0, _int(udp.length) - 8)

        elif hasattr(packet, "icmp"):
            icmp = packet.icmp
            features["protocol"] = "ICMP"
            features["icmp_type"] = _int(icmp.type)
            features["icmp_code"] = _int(icmp.code)
            features["payload_length"] = max(
                0, features["packet_length"] - (_int(ip.hdr_len) if hasattr(ip, "hdr_len") else 20) - 8
            )

        else:
            features["protocol"] = "OTHER"

        return features

    except Exception as exc:
        logger.debug("Feature extraction skipped: %s", exc)
        return None


def infer_app_protocol(protocol: str, dst_port: int, src_port: int) -> str:
    """Return a display-friendly application-layer protocol name.

    Checks dst_port first (server side), then src_port, then falls back to
    the transport protocol string.  Never used for ML — purely for the dashboard.
    """
    if protocol == "ICMP":
        return "ICMP"
    if protocol in ("TCP", "UDP"):
        return _PORT_TO_APP.get(dst_port) or _PORT_TO_APP.get(src_port) or protocol
    return protocol or "OTHER"


def normalize_features(features: dict) -> dict:
    """Normalize a feature dict for ML consumption.

    - Booleans / flag integers are kept as 0/1.
    - Missing keys are filled with 0 (int) or "" (str) from ``_DEFAULTS``.
    - Numeric strings are cast to int/float.
    - Protocol is left as-is (callers can one-hot encode if needed).

    Args:
        features: Raw dict from ``extract_features``.

    Returns:
        New dict with the same keys, all values in plain Python scalars.
    """
    out: dict = {}
    for key in FEATURE_KEYS:
        default = _DEFAULTS[key]
        value = features.get(key, default)

        if isinstance(default, float):
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                out[key] = 0.0
        elif isinstance(default, int):
            try:
                out[key] = int(value)
            except (TypeError, ValueError):
                out[key] = 0
        else:
            out[key] = _str(value) if value is not None else ""

    # Derive application-layer protocol from well-known ports (display only — not used by ML).
    out["app_protocol"] = infer_app_protocol(
        out.get("protocol", "OTHER"),
        out.get("dst_port", 0),
        out.get("src_port", 0),
    )
    return out


def to_ml_vector(features: dict) -> dict:
    """Return the 11-feature dict aligned with CICIDS2017 column names.

    Includes both direct packet-level features and pseudo-flow aggregations
    from the rolling 1-second window. Use this dict for model training and
    inference to guarantee schema consistency with the training data.

    Args:
        features: Raw or normalized dict from ``extract_features``
            (must have been produced with the global FLOW_TRACKER active).

    Returns:
        Dict with keys from ``ML_FEATURE_KEYS`` (CICIDS2017 column names).
    """
    out = {
        cicids_name: features.get(live_name, 0)
        for live_name, cicids_name in CICIDS2017_MAPPING.items()
    }
    # 'Fwd Packets/s' shares the same approximation as 'Flow Packets/s'
    out["Fwd Packets/s"] = features.get("packet_freq_1s", 0.0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import gc
    import logging
    import socket
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from api.capture import live_capture

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    TARGET = 10

    def _resolve(hostname: str) -> list[str]:
        """Return all IPv4 addresses for *hostname*."""
        try:
            infos = socket.getaddrinfo(hostname, 443, socket.AF_INET, socket.SOCK_STREAM)
            return list({info[4][0] for info in infos})
        except socket.gaierror:
            return []

    async def _test_youtube(target: int = TARGET) -> None:
        # Resolve YouTube + Google Video CDN IPs
        hosts = ["youtube.com", "www.youtube.com", "googlevideo.com"]
        youtube_ips: set[str] = set()
        for h in hosts:
            ips = _resolve(h)
            youtube_ips.update(ips)
            if ips:
                print(f"  {h} → {', '.join(ips)}")

        if not youtube_ips:
            print("  ⚠️  Could not resolve YouTube IPs — capturing all port-443 traffic instead")

        # BPF: only TCP port 443 (HTTPS / TLS)
        bpf = "tcp port 443"
        print(f"\n  BPF filter : {bpf}")
        print(f"  Waiting for {target} YouTube packets… (open YouTube in your browser)\n")

        collected: list[dict] = []
        gen = live_capture("eth0", bpf_filter=bpf)
        try:
            async for pkt in gen:
                f = extract_features(pkt)
                if f is None:
                    continue

                # Keep only packets to/from a known YouTube IP (if resolved)
                if youtube_ips and f["src_ip"] not in youtube_ips and f["dst_ip"] not in youtube_ips:
                    continue

                n = normalize_features(f)
                collected.append(n)
                idx = len(collected)

                direction = "→ YT" if n["dst_ip"] in youtube_ips else "← YT"
                print(f"┌─ Packet [{idx}] {direction} {'─' * 35}")
                for key in FEATURE_KEYS:
                    print(f"│  {key:<18} : {n[key]}")
                print(f"└{'─' * 50}")

                if idx >= target:
                    break
        finally:
            await gen.aclose()

        print(f"\n  ✅ Captured {len(collected)} YouTube packet(s)")
        gc.collect()
        await asyncio.sleep(0.2)

    print("🔍 Resolving YouTube IPs…")
    asyncio.run(_test_youtube())
