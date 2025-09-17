#!/usr/bin/env python3
"""
一次性生成 100 个复杂合成 pcap（仅本地回环，IPv4/IPv6 + TCP + TLS + HTTP/2 + WebSocket + protobuf + flag）
保存后运行：sudo python3 gen_clean.py
"""
import os, time, random, struct, uuid
from pathlib import Path
from scapy.all import *
from scapy.layers.inet import TCP, IP, UDP
from scapy.layers.inet6 import IPv6, IPv6ExtHdrFragment
from scapy.layers.dns import DNS, DNSQR, DNSRR
import h2.config, h2.connection

NUM_PCAP = 100
OUT_DIR = Path("../pcap_file/batch_pcap")
OUT_DIR.mkdir(exist_ok=True)

def random_bytes(n):
    return os.urandom(n)

def flag_bytes(i):
    return f"flag{{batch_{i:03d}_{uuid.uuid4().hex[:8]}}}".encode()

def build_proto(flag):
    return b"\x0a" + bytes([len(flag)]) + flag

def ws_frame(payload):
    mask = os.urandom(4)
    header = struct.pack("!BB", 0x82, len(payload) | 0x80)   # 0x82 = binary + mask
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return header + mask + masked

def h2_wrap(payload):
    config = h2.config.H2Configuration(client_side=True)
    conn = h2.connection.H2Connection(config)
    out = [conn.data_to_send()]          # 前导帧
    headers = [
        (':method', 'POST'),
        (':scheme', 'https'),
        (':path', '/ws'),
        (':authority', 'localhost'),
        ('content-type', 'application/octet-stream')
    ]
    conn.send_headers(stream_id=1, headers=headers, end_stream=False)
    conn.send_data(stream_id=1, data=payload, end_stream=True)
    out.append(conn.data_to_send())
    return b''.join(x for x in out if x)

def tls_app_data(inner):
    # 伪 TLS 1.2 Application Data 记录
    return Raw(load=b'\x17\x03\x03' + struct.pack('!H', len(inner)) + inner)

def tcp_mangle_options(tcp):
    opts = [(5, b'\x00\x00\x00\x00'), (8, b'\x00\x00'), (28, b'\x0a\x0b\x0c\x0d')]
    random.shuffle(opts)
    tcp.options = opts
    return tcp

def generate_one(flag, idx):
    packets = []
    # 1. 噪音：DoQ-like
    for _ in range(3):
        noise = IPv6(dst="::1")/UDP(sport=random.randint(5000, 6000), dport=853)/random_bytes(1200)
        noise.time = time.time() + random.uniform(-0.2, 0.2)
        packets.append(noise)

    # 2. TCP 三次握手
    sport, dport = random.randint(30000, 50000), 443
    seq = random.randint(0, 2**32-1)
    ack = random.randint(0, 2**32-1)

    syn = IP(dst="127.0.0.1")/tcp_mangle_options(TCP(sport=sport, dport=dport, flags="S", seq=seq))
    synack = IP(dst="127.0.0.1")/TCP(sport=dport, dport=sport, flags="SA", seq=ack, ack=seq+1)
    ackpkt = IP(dst="127.0.0.1")/TCP(sport=sport, dport=dport, flags="A", seq=seq+1, ack=ack+1)
    for p in (syn, synack, ackpkt):
        p.time = time.time() + random.uniform(0, 0.3)
        packets.append(p)

    # 3. 嵌套 payload
    inner = build_proto(flag)
    ws = ws_frame(inner)
    h2 = h2_wrap(ws)
    tls = tls_app_data(h2)

    # 4. 分段 + 随机重传
    chunk = 500
    for i in range(0, len(tls), chunk):
        seg = IP(dst="127.0.0.1")/TCP(sport=sport, dport=dport, flags="PA",
                                       seq=seq+1+i//chunk*500, ack=ack+1)/tls[i:i+chunk]
        seg.time = time.time() + 0.4 + i*0.001
        packets.append(seg)
        if random.random() < 0.3:
            dup = seg.copy()
            dup.time += 0.01
            packets.append(dup)

    # 5. IPv6 分片镜像
    frag_id = random.randint(0, 2**32-1)
    frag1 = IPv6(dst="::1")/IPv6ExtHdrFragment(nh=6, id=frag_id, offset=0, m=1)/Raw(load=bytes(tls)[:600])
    frag2 = IPv6(dst="::1")/IPv6ExtHdrFragment(nh=6, id=frag_id, offset=75, m=0)/Raw(load=bytes(tls)[600:])
    frag1.time = time.time() + 0.9
    frag2.time = time.time() + 0.95
    packets.extend([frag1, frag2])

    # 6. 随机 RST
    rst = IP(dst="127.0.0.1")/TCP(sport=sport, dport=dport, flags="R", seq=seq+2000)
    rst.time = time.time() + 1.0
    packets.append(rst)

    # 7. 写 pcap
    outfile = OUT_DIR / f"insane_{idx:03d}.pcap"
    wrpcap(str(outfile), sorted(packets, key=lambda p: p.time))
    print(f"[+] {outfile}  done  flag={flag.decode()}")

if __name__ == "__main__":
    for i in range(NUM_PCAP):
        generate_one(flag_bytes(i), i)
    print(f"\n[+] 全部完成 → {OUT_DIR.resolve()}")