# infer.py
import torch
from scapy.all import rdpcap, TCP, UDP, Raw
from model import FlowClassifier  # 使用你训练时的网络结构
import numpy as np
import re
import base64
import zlib
import binascii

PRINT_RE = re.compile(rb'[\x20-\x7e]{6,200}')

def reassemble_flows(pcap_path):
    pkts = rdpcap(pcap_path)
    flows = {}
    for p in pkts:
        if Raw not in p:
            continue
        proto = 6 if TCP in p else (17 if UDP in p else 0)
        try:
            src = p[0][1].src
            dst = p[0][1].dst
        except:
            continue
        sport = getattr(p, 'sport', 0)
        dport = getattr(p, 'dport', 0)
        key = (src, sport, dst, dport, proto)
        revkey = (dst, dport, src, sport, proto)
        if key in flows:
            flows[key].append(bytes(p[Raw].load))
        elif revkey in flows:
            flows[revkey].append(bytes(p[Raw].load))
        else:
            flows[key] = [bytes(p[Raw].load)]
    return [{'five_tuple': k, 'payload': b''.join(v)} for k, v in flows.items()]

def payload_to_tensor(payload, max_len=1024):
    arr = np.frombuffer(payload[:max_len], dtype=np.uint8).astype(np.float32) / 255.0
    if len(arr) < max_len:
        arr = np.pad(arr, (0, max_len - len(arr)))
    return torch.tensor(arr).unsqueeze(0)

def extract_candidates(payload):
    candidates = set()
    for m in PRINT_RE.findall(payload):
        candidates.add(m.decode('utf-8', errors='ignore'))
    try:
        b64 = base64.b64decode(payload, validate=True)
        for m in PRINT_RE.findall(b64):
            candidates.add(m.decode('utf-8', errors='ignore'))
    except: pass
    try:
        s = re.sub(rb'[^0-9a-fA-F]', b'', payload)
        if len(s) >= 4:
            hx = binascii.unhexlify(s)
            for m in PRINT_RE.findall(hx):
                candidates.add(m.decode('utf-8', errors='ignore'))
    except: pass
    try:
        z = zlib.decompress(payload)
        for m in PRINT_RE.findall(z):
            candidates.add(m.decode('utf-8', errors='ignore'))
    except: pass
    return list(candidates)

def infer_pcap(model_path, pcap_path, max_len=1024):
    device = torch.device('cpu')
    model = FlagNet(input_size=max_len)  # 确保和训练时一致
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    flows = reassemble_flows(pcap_path)
    results = []

    for f in flows:
        payload = f['payload']
        if not payload:
            continue
        tensor = payload_to_tensor(payload, max_len=max_len)
        with torch.no_grad():
            out = model(tensor)
            pred_score = out.squeeze().item()
        candidates = extract_candidates(payload)
        results.append({
            'five_tuple': f['five_tuple'],
            'score': pred_score,
            'candidates': candidates
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("用法: python infer.py <model.pth> <input.pcap>")
        sys.exit(1)
    model_path = sys.argv[1]
    pcap_path = sys.argv[2]
    out = infer_pcap(model_path, pcap_path)
    for item in out[:20]:
        print(item)
