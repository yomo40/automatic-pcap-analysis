from scapy.all import *
import random, base64, zlib, time

# ----------------- 基础工具函数 -----------------
def random_ip():
    return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def random_port():
    return random.randint(1024,65535)

def random_flag():
    return f"FLAG{{{random.randint(10000,99999)}_{random.randint(10000,99999)}}}"

def hide_flag(flag, method='plain'):
    if method == 'base64':
        return base64.b64encode(flag.encode())
    elif method == 'zlib':
        return zlib.compress(flag.encode())
    else:
        return flag.encode()

# ----------------- TCP 会话生成 -----------------
def generate_tcp_session(src_ip, dst_ip, src_port, dst_port, flag):
    packets = []
    # 三次握手
    seq_start = random.randint(1000,10000)
    syn = IP(src=src_ip, dst=dst_ip)/TCP(sport=src_port, dport=dst_port, flags='S', seq=seq_start)
    synack = IP(src=dst_ip, dst=src_ip)/TCP(sport=dst_port, dport=src_port, flags='SA', seq=seq_start+1, ack=seq_start+1)
    ack = IP(src=src_ip, dst=dst_ip)/TCP(sport=src_port, dport=dst_port, flags='A', seq=seq_start+1, ack=seq_start+2)
    packets.extend([syn, synack, ack])

    # 数据传输 (随机请求/响应，分片)
    for _ in range(random.randint(2,5)):
        method = random.choice(['plain','base64','zlib'])
        payload = hide_flag(flag, method)
        # 可能分片
        if len(payload) > 20 and random.random() < 0.5:
            chunk_size = random.randint(10,20)
            for i in range(0, len(payload), chunk_size):
                pkt = IP(src=src_ip, dst=dst_ip)/TCP(sport=src_port, dport=dst_port, flags='PA')/Raw(load=payload[i:i+chunk_size])
                packets.append(pkt)
        else:
            pkt = IP(src=src_ip, dst=dst_ip)/TCP(sport=src_port, dport=dst_port, flags='PA')/Raw(load=payload)
            packets.append(pkt)

    # 断开连接
    fin = IP(src=src_ip, dst=dst_ip)/TCP(sport=src_port, dport=dst_port, flags='FA', seq=seq_start+100)
    finack = IP(src=dst_ip, dst=src_ip)/TCP(sport=dst_port, dport=src_port, flags='FA', seq=seq_start+200)
    packets.extend([fin, finack])
    return packets

# ----------------- UDP 包生成 -----------------
def generate_udp_packet(src_ip, dst_ip):
    if random.random() < 0.5:
        # DNS 查询
        qname = f"{random_flag()}.example.com"
        return IP(src=src_ip, dst=dst_ip)/UDP(sport=random_port(), dport=53)/Raw(load=qname.encode())
    else:
        return IP(src=src_ip, dst=dst_ip)/UDP(sport=random_port(), dport=random_port())/Raw(load=random_flag().encode())

# ----------------- ICMP 包生成 -----------------
def generate_icmp_packet(src_ip, dst_ip):
    return IP(src=src_ip, dst=dst_ip)/ICMP()

# ----------------- HTTP 模拟 -----------------
def generate_http_packet(src_ip, dst_ip):
    method = random.choice(['GET','POST'])
    flag = random_flag()
    hide_method = random.choice(['plain','base64','zlib'])
    payload = hide_flag(flag, hide_method)
    if method == 'GET':
        http_payload = f"GET /?q={payload} HTTP/1.1\r\nHost: {dst_ip}\r\nUser-Agent: ProMaxBot\r\n\r\n"
    else:
        http_payload = f"POST /submit HTTP/1.1\r\nHost: {dst_ip}\r\nContent-Length:{len(payload)}\r\n\r\n{payload}"
    return IP(src=src_ip, dst=dst_ip)/TCP(sport=random_port(), dport=80)/Raw(load=http_payload.encode())

# ----------------- 生成单个复杂 PCAP -----------------
def generate_complex_pcap(filename, num_sessions=10):
    packets = []
    for _ in range(num_sessions):
        proto = random.choice(['TCP','UDP','ICMP','HTTP'])
        src_ip, dst_ip = random_ip(), random_ip()
        if proto == 'TCP':
            src_port, dst_port = random_port(), random_port()
            flag = random_flag()
            packets.extend(generate_tcp_session(src_ip, dst_ip, src_port, dst_port, flag))
        elif proto == 'UDP':
            packets.append(generate_udp_packet(src_ip, dst_ip))
        elif proto == 'ICMP':
            packets.append(generate_icmp_packet(src_ip, dst_ip))
        elif proto == 'HTTP':
            packets.append(generate_http_packet(src_ip, dst_ip))
    wrpcap(filename, packets)

# ----------------- 批量生成 -----------------
NUM_FILES = 1000  # 可调整生成数量
for i in range(NUM_FILES):
    generate_complex_pcap(f"promax_pcap_{i}.pcap", num_sessions=random.randint(5,15))
    print(f"[INFO] Generated pro max PCAP {i+1}/{NUM_FILES}")
    time.sleep(0.05)  # 模拟生成间隔
