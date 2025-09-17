#!/usr/bin/env python3
"""
infer_pcap_flag.py
Load the trained 1D-CNN and scan a PCAP file.
"""

import torch, argparse, re
from scapy.all import RawPcapReader, TCP, UDP, IP
from train_pcap_flag import Flag1DCNN, bytes_to_idx

FLAG_REGEX = re.compile(rb'flag\{[^}]+\}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scan_pcap(pcap_path, model, seq_len=1024, batch=512):
    model.eval()
    buf, meta = [], []
    with RawPcapReader(pcap_path) as f:
        for pkt_data, pkt_meta in f:
            if len(pkt_data) < 50: continue
            load = pkt_data[54:]  # 假设以太网+IP+TCP共54B，可调
            if len(load) < 20: continue
            buf.append(load)
            meta.append(pkt_meta)
            if len(buf) == batch:
                yield from infer_batch(buf, meta, model, seq_len)
                buf, meta = [], []
    if buf:
        yield from infer_batch(buf, meta, model, seq_len)

def infer_batch(buf, meta, model, seq_len):
    idx = torch.stack([bytes_to_idx(b, seq_len) for b in buf]).to(device)
    with torch.no_grad():
        scores = model(idx).sigmoid().cpu()
    for i, (s, m) in enumerate(zip(scores, meta)):
        if s > 0.5:
            payload = buf[i]
            # 二次正则确认
            m1 = FLAG_REGEX.search(payload)
            if m1:
                yield m1.group(0).decode(errors='ignore'), m.sec, len(payload)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pcap')
    ap.add_argument('--weights', default='flag_cnn.pt')
    ap.add_argument('--seq_len', type=int, default=1024)
    args = ap.parse_args()

    model = Flag1DCNN(seq_len=args.seq_len).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print('scanning ...')
    for flag, ts, plen in scan_pcap(args.pcap, model, args.seq_len):
        print(f'[+] flag={flag}  ts={ts:.3f}s  len={plen}')

if __name__ == '__main__':
    main()