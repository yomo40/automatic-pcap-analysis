#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_cpu_heavy.py  ——  纯CPU暴力50轮循环训练，吃满所有核心
usage:
    python3 main_cpu_heavy.py --train     # 50轮全量训练
    python3 main_cpu_heavy.py traffic.pcap   # 用最后一轮模型推理
"""
import os, re, json, pickle, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from scapy.all import rdpcap, TCP, UDP, IP, Raw
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

# ---------- 配置 ----------
PCAP_DIR      = Path("./pcap_file")          # 当前目录
MODEL_PATH    = "flag_hunter_cpu50.pkl"
N_ROUND       = 50                 # 循环训练轮数
N_FOLD        = 5                  # 每轮交叉验证折数
HASH_DIM      = 2 ** 18
NGRAM_RANGE   = (1, 4)
FLAG_RE       = re.compile(rb"(flag{[^}]+}|FLAG_[^\s]+|ctf{[^}]+})", re.I)
# --------------------------

# ---------- 通用工具 ----------
class ProtocolOneHot(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        protos = ['tcp', 'udp', 'icmp', 'dns', 'http', 'tls', 'other']
        return np.array([[int(x.get(p, 0)) for p in protos] for x in X])

class StatsFeats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        keys = ['pkts', 'bytes', 'entropy', 'ttl_std', 'comp_ratio']
        return np.array([[x[k] for k in keys] for x in X])

def entropy(b):
    if not b: return 0
    cnt = np.bincount(np.frombuffer(b, dtype=np.uint8))
    cnt = cnt[cnt > 0]
    probs = cnt / len(b)
    return -np.sum(probs * np.log2(probs))

def _get_payload_str(x):
    return x if isinstance(x, str) else x['payload_str']

# ---------- 解析 ----------
def pcap_to_samples(pcap_path: Path):
    samples, labels = [], []
    try:
        pkts = rdpcap(str(pcap_path))
    except Exception as e:
        print(f"[!] skip {pcap_path}: {e}")
        return samples, labels

    flow = defaultdict(bytes)
    for p in pkts:
        if IP in p:
            if TCP in p and Raw in p:
                key = tuple(sorted([(p[IP].src, p[TCP].sport),
                                    (p[IP].dst, p[TCP].dport)]))
                flow[key] += p[Raw].load
            elif UDP in p and Raw in p:
                key = tuple(sorted([(p[IP].src, p[UDP].sport),
                                    (p[IP].dst, p[UDP].dport)]))
                flow[key] += p[Raw].load

    for payload in flow.values():
        if len(payload) < 16: continue
        # 自动标签
        label = 1 if (FLAG_RE.search(payload) or
                      (7.8 < entropy(payload) < 7.99 and len(payload) < 2 * 1024 * 1024 and
                       any(m in payload for m in [b'\x89PNG', b'\xff\xd8', b'PK\x03\x04', b'%PDF']))) else 0
        feat = {
            'tcp': 1,
            'udp': 0,
            'pkts': 0,
            'bytes': len(payload),
            'entropy': entropy(payload),
            'ttl_std': 0,
            'comp_ratio': len(payload) / (len(bz2.compress(payload)) + 1e-6),
            'payload_str': payload[:4096].decode('latin1', 'ignore')
        }
        samples.append(feat)
        labels.append(label)
    return samples, labels

# ---------- 50轮CPU狂练 ----------
def train():
    pcap_files = list(PCAP_DIR.glob("*.pcap"))
    if not pcap_files:
        print("[!] 当前目录无 .pcap 训练文件"); return

    from multiprocessing import Pool

    with Pool() as pool:
        results = list(tqdm(pool.imap(pcap_to_samples, pcap_files), total=len(pcap_files), desc="parse"))

    all_X = sum([r[0] for r in results], [])
    all_y = np.concatenate([r[1] for r in results])
    all_y = np.array(all_y, dtype=int)
    print(f"[*] 总样本 {len(all_X)}  正例 {all_y.sum()}")

    # 构造pipeline
    proto_feat = ProtocolOneHot()
    stats_feat = StatsFeats()
    text_vec   = HashingVectorizer(alternate_sign=False, n_features=HASH_DIM,
                                   ngram_range=NGRAM_RANGE, analyzer='char',
                                   preprocessor=_get_payload_str)
    union = FeatureUnion([
        ('proto', proto_feat),
        ('stats', stats_feat),
        ('text',  text_vec)
    ])

    best_clf, best_score = None, 0.0
    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=42)

    for rnd in range(1, N_ROUND + 1):
        print(f"\n========== Round {rnd}/{N_ROUND} ==========")
        round_t0 = time.time()

        # 每轮重新训练全新模型（CPU拉满）
        from sklearn.svm import LinearSVC

        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=500,  # 更多树 = 更多并行
            max_depth=None,
            n_jobs=-1,  # 占满所有核心
            verbose=0
        )

        pipe = Pipeline([('feat', union),
                         ('scaler', StandardScaler(with_mean=False)),
                         ('clf', clf)])

        fold_scores = []
        from joblib import Parallel, delayed

        def fit_fold(train_idx, val_idx):
            X_train = [all_X[i] for i in train_idx]
            y_train = all_y[train_idx]
            X_val = [all_X[i] for i in val_idx]
            y_val = all_y[val_idx]
            pipe.fit(X_train, y_train)
            return pipe.score(X_val, y_val)

        fold_scores = Parallel(n_jobs=-1)(delayed(fit_fold)(train_idx, val_idx)
                                          for train_idx, val_idx in skf.split(all_X, all_y))

        avg_score = np.mean(fold_scores)
        print(f"CV accuracy: {avg_score:.4f}  time: {time.time()-round_t0:.1f}s")
        if avg_score > best_score:
            best_score, best_clf = avg_score, pipe

    # 保存最后一轮（也可改成 best_clf）
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_clf, f)
    print(f"[+] 50轮训练完成，模型保存 -> {MODEL_PATH}  best_CV={best_score:.4f}")

# ---------- 推理 ----------
def inference(pcap_path: Path):
    if not os.path.exists(MODEL_PATH):
        print("[!] 找不到模型，先 --train"); return
    with open(MODEL_PATH, 'rb') as f:
        pipe = pickle.load(f)

    samples, _ = pcap_to_samples(pcap_path)
    if not samples:
        print("[*] 无流量可分析"); return
    preds = pipe.predict(samples)
    flags = set()
    for feat, p in zip(samples, preds):
        if p == 1:
            payload = feat['payload_str'].encode('latin1')
            flags.update(re.findall(FLAG_RE, payload))
    print(json.dumps([f.decode(errors='ignore') for f in flags], indent=2))

# ---------- CLI ----------
if __name__ == '__main__':
    import argparse, bz2
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs='?', help="待推理 pcap")
    parser.add_argument("--train", action="store_true", help="50轮CPU训练")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.file:
        inference(Path(args.file))
    else:
        parser.print_help()