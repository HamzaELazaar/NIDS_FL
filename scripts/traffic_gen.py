#!/usr/bin/env python3
"""
traffic_gen.py — Synthetic network traffic generator for NIDS pipeline testing.

Two modes
---------
  default  : generates real HTTP / ICMP / TCP traffic on the network so pyshark
             captures it on the configured interface (CAPTURE_IFACE, default eth0).

  --direct : POSTs synthetic feature vectors straight to the API's /inject
             endpoint — works even when pyshark is not capturing.
             This is the recommended mode for dashboard testing.

Usage:
    python scripts/traffic_gen.py --direct                          # mixed traffic, 5% attacks
    python scripts/traffic_gen.py --direct --attack-type DDoS       # 100% DDoS packets
    python scripts/traffic_gen.py --direct --attack-type PortScan   # 100% port scan
    python scripts/traffic_gen.py --direct --attack-rate 0.4        # 40% random attacks
    python scripts/traffic_gen.py --direct --attack-type mixed --attack-rate 0.6
    python scripts/traffic_gen.py --direct --duration 120 --intensity high

Attack types: DDoS | DoS | PortScan | BruteForce | Bots | WebAttacks | mixed | normal
"""

import argparse
import json
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

API_BASE = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
# Target lists (real network mode)
# ─────────────────────────────────────────────────────────────────────────────

HTTP_TARGETS = [
    "http://httpbin.org/get",     "http://example.com",
    "http://neverssl.com",        "http://httpforever.com",
    "http://detectportal.firefox.com/success.txt",
]
HTTPS_TARGETS = [
    "https://www.google.com",     "https://cloudflare.com",
    "https://httpbin.org/get",    "https://api.github.com",
    "https://jsonplaceholder.typicode.com/posts/1",
]
PING_HOSTS   = ["8.8.8.8", "1.1.1.1", "8.8.4.4"]
TCP_PROBES   = [("8.8.8.8", 53), ("www.google.com", 80), ("github.com", 443)]

# ─────────────────────────────────────────────────────────────────────────────
# IP helpers
# ─────────────────────────────────────────────────────────────────────────────

_PRIVATE_RANGES = [("192.168.1.", range(2, 254)), ("10.0.0.", range(2, 254))]
_PUBLIC_IPS = ["8.8.8.8", "1.1.1.1", "142.250.80.46", "104.21.64.1", "93.184.216.34"]


def _rip() -> str:
    prefix, rng = random.choice(_PRIVATE_RANGES)
    return prefix + str(random.choice(list(rng)))


def _val(v):
    """Resolve a value that may be a callable (lambda) or a plain scalar."""
    return v() if callable(v) else v


# ─────────────────────────────────────────────────────────────────────────────
# Attack profiles — feature vectors calibrated to CICIDS2017 distributions
#
# 11 ML features used by MLP + XGBoost:
#   dst_port, fin_flag, psh_flag, ack_flag, window_size, payload_length,
#   packet_count_1s, bytes_per_sec_1s, avg_packet_size_1s, packet_freq_1s
# ─────────────────────────────────────────────────────────────────────────────

# Each profile value can be a plain scalar or a zero-argument callable (lambda).

_PROFILES: dict[str, dict] = {

    # ── Normal traffic ────────────────────────────────────────────────────────
    # Flags use realistic CICIDS2017 TCP session-state weights, not independent
    # coin flips.  The ML model uses fin/psh/ack — random combos produce
    # impossible states (FIN=1,PSH=1,ACK=0) that the model never saw in training.
    #
    # State weights:  PSH+ACK (data)  ACK-only  FIN+ACK (teardown)  no-flags (UDP/ICMP)
    #                    ~62%           ~26%          ~7%                ~5%
    "Normal": {
        "protocol":          lambda: random.choices(["TCP", "UDP", "ICMP"], weights=[0.70, 0.20, 0.10])[0],
        "dst_port":          lambda: random.choice([80, 443, 53, 22, 8080]),
        "packet_length":     lambda: random.randint(64, 1460),
        "ttl":               lambda: random.choice([64, 128]),
        "syn_flag":          0,   # not used by ML model
        "fin_flag":          lambda: random.choices([0, 1], weights=[0.93, 0.07])[0],
        "rst_flag":          0,
        "psh_flag":          lambda: random.choices([0, 1], weights=[0.38, 0.62])[0],
        "ack_flag":          lambda: random.choices([0, 1], weights=[0.12, 0.88])[0],
        "urg_flag":          0,
        "window_size":       lambda: random.choice([8192, 16384, 32768, 65535]),
        "payload_length":    lambda: random.randint(40, 1400),
        "packet_count_1s":   lambda: random.randint(1, 8),
        "packet_freq_1s":    lambda: random.uniform(1, 8),
        "bytes_per_sec_1s":  lambda: random.uniform(200, 6000),
        "avg_packet_size_1s":lambda: random.uniform(200, 1000),
    },

    # ── DDoS — volumetric flood (LOIC / HOIC style) ───────────────────────────
    # High pps, tiny packets, SYN or ACK flood, zero or tiny window
    "DDoS": {
        "protocol":          "TCP",
        "dst_port":          lambda: random.choice([80, 443, 53, random.randint(1024, 65535)]),
        "packet_length":     lambda: random.randint(40, 80),
        "ttl":               lambda: random.choice([64, 128]),
        "syn_flag":          1,
        "fin_flag":          0,
        "rst_flag":          0,
        "psh_flag":          0,
        "ack_flag":          0,
        "urg_flag":          0,
        "window_size":       lambda: random.choice([0, 512, 1024]),
        "payload_length":    lambda: random.randint(0, 20),
        "packet_count_1s":   lambda: random.randint(150, 600),
        "packet_freq_1s":    lambda: random.uniform(150, 600),
        "bytes_per_sec_1s":  lambda: random.uniform(60000, 500000),
        "avg_packet_size_1s":lambda: random.uniform(40, 80),
    },

    # ── DoS — application-layer flood (Hulk / GoldenEye / Slowloris) ─────────
    # Large HTTP payloads, PSH+ACK, moderate rate, port 80/8080
    "DoS": {
        "protocol":          "TCP",
        "dst_port":          lambda: random.choice([80, 443, 8080]),
        "packet_length":     lambda: random.randint(500, 1500),
        "ttl":               64,
        "syn_flag":          0,
        "fin_flag":          0,
        "rst_flag":          0,
        "psh_flag":          1,
        "ack_flag":          1,
        "urg_flag":          0,
        "window_size":       lambda: random.choice([8192, 16384]),
        "payload_length":    lambda: random.randint(400, 1400),
        "packet_count_1s":   lambda: random.randint(60, 250),
        "packet_freq_1s":    lambda: random.uniform(60, 250),
        "bytes_per_sec_1s":  lambda: random.uniform(30000, 300000),
        "avg_packet_size_1s":lambda: random.uniform(500, 1500),
    },

    # ── Port Scanning — SYN scan across many ports ────────────────────────────
    # Tiny SYN packets, ephemeral dst_port, no payload
    "PortScan": {
        "protocol":          "TCP",
        "dst_port":          lambda: random.randint(1, 65535),
        "packet_length":     lambda: random.randint(40, 64),
        "ttl":               lambda: random.choice([64, 128]),
        "syn_flag":          1,
        "fin_flag":          0,
        "rst_flag":          0,
        "psh_flag":          0,
        "ack_flag":          0,
        "urg_flag":          0,
        "window_size":       lambda: random.choice([1024, 2048]),
        "payload_length":    0,
        "packet_count_1s":   lambda: random.randint(15, 80),
        "packet_freq_1s":    lambda: random.uniform(15, 80),
        "bytes_per_sec_1s":  lambda: random.uniform(600, 5000),
        "avg_packet_size_1s":lambda: random.uniform(40, 64),
    },

    # ── Brute Force — SSH / FTP credential stuffing ───────────────────────────
    # Repeated TCP to well-known auth ports, small payloads, rapid reconnects
    "BruteForce": {
        "protocol":          "TCP",
        "dst_port":          lambda: random.choice([22, 21, 3389, 23, 5900]),
        "packet_length":     lambda: random.randint(60, 220),
        "ttl":               64,
        "syn_flag":          lambda: random.choice([0, 1]),
        "fin_flag":          0,
        "rst_flag":          0,
        "psh_flag":          1,
        "ack_flag":          1,
        "urg_flag":          0,
        "window_size":       lambda: random.choice([8192, 65535]),
        "payload_length":    lambda: random.randint(40, 180),
        "packet_count_1s":   lambda: random.randint(8, 40),
        "packet_freq_1s":    lambda: random.uniform(8, 40),
        "bytes_per_sec_1s":  lambda: random.uniform(500, 8000),
        "avg_packet_size_1s":lambda: random.uniform(60, 220),
    },

    # ── Bots — C2 beaconing / lateral movement ────────────────────────────────
    # Periodic low-rate traffic, unusual ports (IRC/C2), variable protocols
    "Bots": {
        "protocol":          lambda: random.choice(["TCP", "UDP"]),
        "dst_port":          lambda: random.choice([6667, 4444, 1234, 8888,
                                                    random.randint(49152, 65535)]),
        "packet_length":     lambda: random.randint(80, 900),
        "ttl":               lambda: random.choice([64, 128]),
        "syn_flag":          0,
        "fin_flag":          0,
        "rst_flag":          0,
        "psh_flag":          lambda: random.randint(0, 1),
        "ack_flag":          1,
        "urg_flag":          0,
        "window_size":       lambda: random.choice([8192, 16384, 32768]),
        "payload_length":    lambda: random.randint(60, 850),
        "packet_count_1s":   lambda: random.randint(2, 20),
        "packet_freq_1s":    lambda: random.uniform(2, 20),
        "bytes_per_sec_1s":  lambda: random.uniform(800, 18000),
        "avg_packet_size_1s":lambda: random.uniform(80, 900),
    },

    # ── Web Attacks — SQLi / XSS / LFI via HTTP ──────────────────────────────
    # Large HTTP POST/GET payloads, PSH+ACK, port 80/443, low-medium rate
    "WebAttacks": {
        "protocol":          "TCP",
        "dst_port":          lambda: random.choice([80, 443, 8080, 8443]),
        "packet_length":     lambda: random.randint(300, 1500),
        "ttl":               64,
        "syn_flag":          0,
        "fin_flag":          0,
        "rst_flag":          0,
        "psh_flag":          1,
        "ack_flag":          1,
        "urg_flag":          0,
        "window_size":       lambda: random.choice([32768, 65535]),
        "payload_length":    lambda: random.randint(250, 1450),
        "packet_count_1s":   lambda: random.randint(1, 12),
        "packet_freq_1s":    lambda: random.uniform(1, 12),
        "bytes_per_sec_1s":  lambda: random.uniform(300, 18000),
        "avg_packet_size_1s":lambda: random.uniform(300, 1500),
    },
}

# Human-readable aliases for CLI
_ALIAS: dict[str, str] = {
    "ddos":        "DDoS",
    "dos":         "DoS",
    "portscan":    "PortScan",
    "port-scan":   "PortScan",
    "portsanning": "PortScan",
    "bruteforce":  "BruteForce",
    "brute-force": "BruteForce",
    "bots":        "Bots",
    "bot":         "Bots",
    "webattacks":  "WebAttacks",
    "web-attacks": "WebAttacks",
    "web":         "WebAttacks",
    "normal":      "Normal",
    "mixed":       "mixed",
}

# Weight distribution for --attack-type mixed
_MIXED_WEIGHTS: dict[str, float] = {
    "DDoS":       0.20,
    "DoS":        0.20,
    "PortScan":   0.20,
    "BruteForce": 0.15,
    "Bots":       0.10,
    "WebAttacks": 0.15,
}


def _pick_attack_type() -> str:
    types, weights = zip(*_MIXED_WEIGHTS.items())
    return random.choices(types, weights=weights, k=1)[0]  # type: ignore


def _make_packet(attack_type: str | None, attack_rate: float) -> dict:
    """Build a feature vector for /inject.

    Args:
        attack_type: None/'mixed' for probabilistic mix, or a specific key.
        attack_rate: 0-1 probability of generating an attack packet
                     (ignored when attack_type is a specific non-normal type).
    """
    if attack_type == "Normal":
        profile_key = "Normal"
    elif attack_type and attack_type != "mixed":
        profile_key = attack_type
    else:
        if random.random() < attack_rate:
            profile_key = _pick_attack_type()
        else:
            profile_key = "Normal"

    p = _PROFILES[profile_key]
    pkt_len  = _val(p["packet_length"])
    pkt_cnt  = _val(p["packet_count_1s"])
    freq     = _val(p["packet_freq_1s"])
    bps      = _val(p["bytes_per_sec_1s"])
    avg_size = _val(p["avg_packet_size_1s"])

    return {
        "timestamp":          time.time(),
        "protocol":           _val(p["protocol"]),
        "src_ip":             _rip(),
        "dst_ip":             random.choice(_PUBLIC_IPS),
        "src_port":           random.randint(1024, 65535),
        "dst_port":           _val(p["dst_port"]),
        "packet_length":      pkt_len,
        "ttl":                _val(p["ttl"]),
        "syn_flag":           _val(p["syn_flag"]),
        "fin_flag":           _val(p["fin_flag"]),
        "rst_flag":           _val(p["rst_flag"]),
        "psh_flag":           _val(p["psh_flag"]),
        "ack_flag":           _val(p["ack_flag"]),
        "urg_flag":           _val(p["urg_flag"]),
        "window_size":        _val(p["window_size"]),
        "icmp_type":          0,
        "icmp_code":          0,
        "payload_length":     _val(p["payload_length"]),
        "packet_count_1s":    pkt_cnt,
        "bytes_per_sec_1s":   round(bps, 2),
        "avg_packet_size_1s": round(avg_size, 2),
        "packet_freq_1s":     round(freq, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared counters
# ─────────────────────────────────────────────────────────────────────────────

_lock   = threading.Lock()
_counts: dict[str, int] = {
    "inject": 0, "err": 0,
    "http": 0, "https": 0, "ping": 0, "tcp": 0,
    **{k: 0 for k in _PROFILES},
}


def _inc(key: str) -> None:
    with _lock:
        _counts[key] = _counts.get(key, 0) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Workers — real network mode
# ─────────────────────────────────────────────────────────────────────────────

def do_http(url: str) -> None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "nids-traffic-gen/1.0"})
        with urllib.request.urlopen(req, timeout=4): pass
        _inc("https" if url.startswith("https") else "http")
    except Exception:
        _inc("err")


def do_ping(host: str, count: int = 4) -> None:
    try:
        subprocess.run(["ping", "-c", str(count), "-W", "2", host],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=count * 3)
        _inc("ping")
    except Exception:
        _inc("err")


def do_tcp_probe(host: str, port: int) -> None:
    import socket
    try:
        with socket.create_connection((host, port), timeout=3): pass
        _inc("tcp")
    except Exception:
        _inc("err")


# ─────────────────────────────────────────────────────────────────────────────
# Worker — direct inject mode
# ─────────────────────────────────────────────────────────────────────────────

def do_inject(attack_type: str | None, attack_rate: float) -> None:
    pkt  = _make_packet(attack_type, attack_rate)
    data = json.dumps(pkt).encode()
    req  = urllib.request.Request(
        f"{API_BASE}/inject", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5): pass
        _inc("inject")
    except Exception:
        _inc("err")


# ─────────────────────────────────────────────────────────────────────────────
# Progress
# ─────────────────────────────────────────────────────────────────────────────

def _progress_loop(start: float, duration: float, direct: bool, attack_type: str | None) -> None:
    bar_w = 36
    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            break
        frac   = min(elapsed / duration, 1.0)
        bar    = "█" * int(bar_w * frac) + "░" * (bar_w - int(bar_w * frac))
        with _lock:
            c = dict(_counts)

        if direct:
            attack_keys = [k for k in _PROFILES if k != "Normal"]
            attack_total = sum(c.get(k, 0) for k in attack_keys)
            line = (
                f"\r  [{bar}] {frac*100:5.1f}%  "
                f"injected={c['inject']}  attacks={attack_total}  err={c['err']}   "
            )
        else:
            line = (
                f"\r  [{bar}] {frac*100:5.1f}%  "
                f"http={c['http']} https={c['https']} ping={c['ping']} tcp={c['tcp']} err={c['err']}   "
            )
        sys.stdout.write(line)
        sys.stdout.flush()
        time.sleep(0.4)
    sys.stdout.write("\n")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NIDS traffic generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Attack types:",
            "  DDoS       — volumetric SYN flood (high pps, tiny packets)",
            "  DoS        — application-layer flood (Hulk/GoldenEye style)",
            "  PortScan   — SYN scan across random destination ports",
            "  BruteForce — credential stuffing on SSH/FTP/RDP ports",
            "  Bots       — C2 beaconing / IRC bot traffic",
            "  WebAttacks — SQLi / XSS / LFI via HTTP POST",
            "  mixed      — weighted random mix of all attack types",
            "  normal     — benign traffic only",
        ]),
    )
    parser.add_argument("--duration",    type=int,   default=60,
                        help="Test duration in seconds (default 60)")
    parser.add_argument("--intensity",   choices=["low", "medium", "high"], default="medium",
                        help="Traffic intensity (default medium)")
    parser.add_argument("--direct",      action="store_true",
                        help="Inject packets directly to /inject (no pyshark needed)")
    parser.add_argument("--attack-type", type=str,   default="mixed",
                        metavar="TYPE",
                        help="Attack type or 'mixed'/'normal' (default: mixed)")
    parser.add_argument("--attack-rate", type=float, default=0.30,
                        metavar="0-1",
                        help="Fraction of injected packets that are attacks (default 0.30)")
    args = parser.parse_args()

    # Resolve alias
    canonical = _ALIAS.get(args.attack_type.lower())
    if canonical is None:
        print(f"Unknown attack type '{args.attack_type}'. "
              f"Choose from: {', '.join(_PROFILES)} mixed normal")
        sys.exit(1)
    attack_type = None if canonical == "mixed" else canonical

    workers  = {"low": 2,  "medium": 5,  "high": 12}[args.intensity]
    interval = {"low": 1.5,"medium": 0.8,"high": 0.3}[args.intensity] if args.direct \
               else {"low": 4, "medium": 2, "high": 1}[args.intensity]
    burst    = {"low": 2,  "medium": 5,  "high": 12}[args.intensity]

    mode_str = "DIRECT INJECT → /inject" if args.direct else "REAL NETWORK (pyshark)"
    type_str = (f"mixed ({args.attack_rate*100:.0f}% attacks)"
                if attack_type is None
                else f"{attack_type} (100%)" if attack_type != "Normal"
                else "Normal (0% attacks)")

    print(f"\n  NIDS Traffic Generator")
    print(f"  {'─'*54}")
    print(f"  Duration  : {args.duration}s  |  Intensity : {args.intensity}  |  Workers : {workers}")
    print(f"  Mode      : {mode_str}")
    print(f"  Traffic   : {type_str}")
    if args.direct:
        print(f"  API       : {API_BASE}")
    print(f"  {'─'*54}\n")

    # Health check
    try:
        with urllib.request.urlopen(f"{API_BASE}/health", timeout=3) as r:
            health = json.loads(r.read())
        pipeline_ok = health.get("pipeline_ready", False)
        print(f"  API health : {health.get('status', '?')}  "
              f"pipeline={pipeline_ok}  "
              f"ws_clients={health.get('ws_clients', 0)}")
        if args.direct and not pipeline_ok:
            print("  ⚠  Pipeline not ready — packets counted but not classified")
    except Exception as exc:
        print(f"  ✗ Cannot reach API at {API_BASE}: {exc}")
        print("    Start the API first:  uvicorn api.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    print()

    start = time.time()
    end   = start + args.duration

    threading.Thread(
        target=_progress_loop,
        args=(start, args.duration, args.direct, attack_type),
        daemon=True,
    ).start()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while time.time() < end:
            if args.direct:
                for _ in range(burst):
                    pool.submit(do_inject, attack_type, args.attack_rate)
            else:
                pool.submit(do_http, random.choice(HTTP_TARGETS + HTTPS_TARGETS))
                for h in PING_HOSTS:
                    pool.submit(do_ping, h)
                for h, p in TCP_PROBES:
                    pool.submit(do_tcp_probe, h, p)

            if time.time() + interval < end:
                time.sleep(interval)

    time.sleep(0.5)  # let progress bar finish

    with _lock:
        c = dict(_counts)

    attack_keys = [k for k in _PROFILES if k != "Normal"]
    attack_total = sum(c.get(k, 0) for k in attack_keys)

    print(f"\n  {'─'*54}")
    if args.direct:
        print(f"  Done.  Injected : {c['inject']}  Errors : {c['err']}")
        if attack_type is None or attack_type != "Normal":
            print(f"\n  Attack breakdown:")
            for k in attack_keys:
                n = c.get(k, 0)
                if n:
                    bar = "▓" * min(n // 2, 30)
                    print(f"    {k:<14} {n:>5}  {bar}")
    else:
        total = c["http"] + c["https"] + c["ping"] + c["tcp"]
        print(f"  Done.  Total : {total}  "
              f"(http={c['http']} https={c['https']} ping={c['ping']} "
              f"tcp={c['tcp']} err={c['err']})")
    print(f"  {'─'*54}\n")


if __name__ == "__main__":
    main()
