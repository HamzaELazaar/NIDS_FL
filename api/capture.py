"""Live packet capture using Python AF_PACKET raw sockets.

No tshark / pyshark / wireshark required.
Uses the kernel's packet socket interface directly — the same path tcpdump uses.

Requirements
------------
* Linux only (AF_PACKET is Linux-specific).
* CAP_NET_RAW capability or root:
    sudo setcap cap_net_raw+ep $(which python3)
  OR run uvicorn with sudo.

The async generator live_capture() yields one feature dict per IP packet,
already populated with pseudo-flow statistics from the shared FlowTracker.
The dict schema is identical to what api.features.extract_features() produced
for pyshark packets, so the rest of the pipeline is unchanged.
"""

import asyncio
import logging
import socket
import struct
import time
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Protocol constants
# ─────────────────────────────────────────────────────────────────────────────

_ETH_P_ALL  = 0x0003   # capture everything (like tcpdump default)
_ETH_P_IP   = 0x0800   # IPv4 ethertype

_PROTO_ICMP = 1
_PROTO_TCP  = 6
_PROTO_UDP  = 17

# ─────────────────────────────────────────────────────────────────────────────
# Packet parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse(raw: bytes, ts: float) -> Optional[dict]:
    """Parse one raw Ethernet frame into a feature dict.

    Returns None for non-IPv4, non-TCP/UDP/ICMP, or malformed frames.
    The returned dict has the same keys as api.features.FEATURE_KEYS
    (minus the pseudo-flow fields which are filled by the caller).
    """
    # Minimum: Ethernet (14) + IP (20)
    if len(raw) < 34:
        return None

    # ── Ethernet header ───────────────────────────────────────────────────────
    ethertype = struct.unpack_from("!H", raw, 12)[0]
    if ethertype != _ETH_P_IP:
        return None

    ip_off = 14

    # ── IPv4 header ───────────────────────────────────────────────────────────
    if len(raw) < ip_off + 20:
        return None

    ihl       = (raw[ip_off] & 0x0F) * 4          # header length in bytes
    ttl       = raw[ip_off + 8]
    proto     = raw[ip_off + 9]
    total_len = struct.unpack_from("!H", raw, ip_off + 2)[0]
    src_ip    = socket.inet_ntoa(raw[ip_off + 12 : ip_off + 16])
    dst_ip    = socket.inet_ntoa(raw[ip_off + 16 : ip_off + 20])

    if ihl < 20 or len(raw) < ip_off + ihl:
        return None

    t_off = ip_off + ihl  # start of transport header

    f: dict = {
        "timestamp":      ts,
        "src_ip":         src_ip,
        "dst_ip":         dst_ip,
        "ttl":            ttl,
        "packet_length":  total_len,
        "protocol":       "OTHER",
        "src_port":       0,
        "dst_port":       0,
        "syn_flag":       0,
        "fin_flag":       0,
        "rst_flag":       0,
        "psh_flag":       0,
        "ack_flag":       0,
        "urg_flag":       0,
        "window_size":    0,
        "icmp_type":      0,
        "icmp_code":      0,
        "payload_length": 0,
    }

    # ── TCP ───────────────────────────────────────────────────────────────────
    if proto == _PROTO_TCP:
        if len(raw) < t_off + 20:
            return None
        src_port, dst_port = struct.unpack_from("!HH", raw, t_off)
        tcp_hdr_len        = ((raw[t_off + 12] >> 4) & 0xF) * 4
        # flags are the low 9 bits of the 2-byte field at offset 12
        flags_word         = struct.unpack_from("!H", raw, t_off + 12)[0] & 0x01FF
        window             = struct.unpack_from("!H", raw, t_off + 14)[0]

        f["protocol"]       = "TCP"
        f["src_port"]       = src_port
        f["dst_port"]       = dst_port
        f["window_size"]    = window
        f["fin_flag"]       = (flags_word >> 0) & 1
        f["syn_flag"]       = (flags_word >> 1) & 1
        f["rst_flag"]       = (flags_word >> 2) & 1
        f["psh_flag"]       = (flags_word >> 3) & 1
        f["ack_flag"]       = (flags_word >> 4) & 1
        f["urg_flag"]       = (flags_word >> 5) & 1
        f["payload_length"] = max(0, total_len - ihl - tcp_hdr_len)

    # ── UDP ───────────────────────────────────────────────────────────────────
    elif proto == _PROTO_UDP:
        if len(raw) < t_off + 8:
            return None
        src_port, dst_port, udp_len = struct.unpack_from("!HHH", raw, t_off)

        f["protocol"]       = "UDP"
        f["src_port"]       = src_port
        f["dst_port"]       = dst_port
        f["payload_length"] = max(0, udp_len - 8)

    # ── ICMP ──────────────────────────────────────────────────────────────────
    elif proto == _PROTO_ICMP:
        if len(raw) < t_off + 4:
            return None
        f["protocol"]       = "ICMP"
        f["icmp_type"]      = raw[t_off]
        f["icmp_code"]      = raw[t_off + 1]
        f["payload_length"] = max(0, total_len - ihl - 8)

    else:
        return None  # skip anything that isn't TCP / UDP / ICMP

    return f


# ─────────────────────────────────────────────────────────────────────────────
# Async capture generator
# ─────────────────────────────────────────────────────────────────────────────

async def live_capture(
    interface: str = "eth0",
    bpf_filter: str = "",          # kept for API compatibility, unused here
) -> AsyncGenerator[dict, None]:
    """Async generator that yields one feature dict per captured IP packet.

    Uses AF_PACKET raw sockets — no external tools required.

    Args:
        interface: Network interface name, e.g. ``"eth0"``.
        bpf_filter: Ignored (kept so callers don't need changing).

    Yields:
        Feature dicts with keys matching ``api.features.FEATURE_KEYS``,
        including pseudo-flow aggregations from the shared FlowTracker.

    Raises:
        RuntimeError: Permission denied or interface not found.
    """
    from api.features import FLOW_TRACKER  # local import avoids circular dep

    try:
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(_ETH_P_ALL))
    except PermissionError:
        raise RuntimeError(
            f"Permission denied opening raw socket on '{interface}'.\n"
            "Grant CAP_NET_RAW to the Python interpreter:\n"
            "  sudo setcap cap_net_raw+ep $(which python3)\n"
            "or run uvicorn with sudo."
        )
    except OSError as exc:
        raise RuntimeError(f"Cannot open raw socket: {exc}") from exc

    try:
        sock.bind((interface, 0))
    except OSError as exc:
        sock.close()
        raise RuntimeError(
            f"Cannot bind to '{interface}': {exc}\n"
            f"Available: {', '.join(n['name'] for n in list_interfaces())}"
        ) from exc

    sock.setblocking(False)
    loop = asyncio.get_event_loop()
    logger.info("Raw socket capture started on %s (no tshark)", interface)

    try:
        while True:
            try:
                raw = await loop.sock_recv(sock, 65536)
            except asyncio.CancelledError:
                break
            except OSError as exc:
                logger.error("Socket recv error: %s", exc)
                break

            ts       = time.time()
            features = _parse(raw, ts)
            if features is None:
                continue

            # Attach pseudo-flow stats (rolling 1-second window per src_ip)
            pseudo = FLOW_TRACKER.update(
                src_ip=features["src_ip"],
                timestamp=ts,
                packet_length=features["packet_length"],
            )
            features.update(pseudo)
            yield features

    finally:
        sock.close()
        logger.info("Raw socket capture stopped on %s", interface)


# ─────────────────────────────────────────────────────────────────────────────
# Interface listing (no tshark needed)
# ─────────────────────────────────────────────────────────────────────────────

def list_interfaces() -> list[dict[str, str]]:
    """Return network interfaces from /proc/net/dev (Linux, no external tools)."""
    interfaces: list[dict[str, str]] = []
    try:
        with open("/proc/net/dev") as fh:
            for i, line in enumerate(fh):
                if i < 2:
                    continue  # skip the two header lines
                name = line.strip().split(":")[0].strip()
                if name:
                    interfaces.append({"index": str(i - 1), "name": name})
    except OSError:
        pass
    return interfaces


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("Interfaces found in /proc/net/dev:")
    for iface in list_interfaces():
        print(f"  [{iface['index']}] {iface['name']}")

    iface = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    print(f"\nCapturing {count} packets on {iface} …")

    async def _test() -> None:
        n = 0
        async for f in live_capture(iface):
            n += 1
            print(f"  [{n:>3}] {f['protocol']:<5} {f['src_ip']}:{f['src_port']} "
                  f"→ {f['dst_ip']}:{f['dst_port']}  len={f['packet_length']}")
            if n >= count:
                break

    asyncio.run(_test())
