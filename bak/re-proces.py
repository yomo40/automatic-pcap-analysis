# requirements:
# pip install scapy numpy pandas scikit-learn lightgbm joblib zlib

import os, re, math, zlib, base64
from scapy.all import rdpcap, TCP, UDP, Raw
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import binascii

# ----------------- 工具: 流重组（按五元组聚合） -----------------
def reassemble_flows(pcap_path):
    pkts = rdpcap(pcap_path)
    flows = {}  # key: (src, sport, dst, dport, proto) normalized
    for p in pkts:
        if Raw not in p:
            continue
        proto = 6 if TCP in p else (17 if UDP in p else 0)
        # 尽量获取端口和地址字段，scapy 对某些 pcap 可能不同
        try:
            src = p[0][1].src
            dst = p[0][1].dst
        except:
            continue
        sport = p.sport if hasattr(p, 'sport') else (p.sport if hasattr(p, 'sport') else 0)
        dport = p.dport if hasattr(p, 'dport') else (p.dport if hasattr(p, 'dport') else 0)
        # 归一化方向（小->大）
        key = (src, sport, dst, dport, proto)
        revkey = (dst, dport, src, sport, proto)
        if key in flows:
            flows[key].append(bytes(p[Raw].load))
        elif revkey in flows:
            flows[revkey].append(bytes(p[Raw].load))
        else:
            flows[key] = [bytes(p[Raw].load)]
    # 合并方向数据为流 payload（按顺序拼接）
    flow_payloads = []
    for k, segs in flows.items():
        flow_payloads.append({
            'five_tuple': k,
            'payload': b''.join(segs)
        })
    return flow_payloads

# ----------------- 自动解码尝试（base64/hex/xor/rot/zlib） -----------------
def try_base64(b):
    try:
        return base64.b64decode(b, validate=True)
    except Exception:
        return None

def try_hex(b):
    s = re.sub(rb'[^0-9a-fA-F]', b'', b)
    if len(s) < 4: return None
    try:
        return binascii.unhexlify(s)
    except Exception:
        return None

def try_zlib(b):
    try:
        return zlib.decompress(b)
    except Exception:
        return None

def try_rot(payload, maxrot=13):
    out = []
    for r in range(1, maxrot+1):
        try:
            dec = bytes(( (c - r) & 0xff ) for c in payload)
            out.append(dec)
        except Exception:
            pass
    return out

def try_xor_small_keys(payload, max_key=255):
    # 试短 key 或 1字节异或
    out = []
    if not payload: return out
    for k in range(1, 64):  # 仅尝试 1..63 避免爆炸
        try:
            dec = bytes([c ^ k for c in payload])
            out.append((k, dec))
        except Exception:
            pass
    return out

# ----------------- 候选抽取: 滑窗 + 解码 + 可打印子串 -----------------
PRINT_RE = re.compile(rb'[\x20-\x7e]{6,200}')  # 可调整长度
def extract_candidates_from_payload(payload):
    candidates = set()
    # 1) 直接可打印子串
    for m in PRINT_RE.findall(payload):
        candidates.add(m.decode('utf-8', errors='ignore'))
    # 2) base64/hex/zlib try
    b64 = try_base64(payload)
    if b64:
        for m in PRINT_RE.findall(b64):
            candidates.add(m.decode('utf-8', errors='ignore'))
    hx = try_hex(payload)
    if hx:
        for m in PRINT_RE.findall(hx):
            candidates.add(m.decode('utf-8', errors='ignore'))
    z = try_zlib(payload)
    if z:
        for m in PRINT_RE.findall(z):
            candidates.add(m.decode('utf-8', errors='ignore'))
    # 3) xor small keys and rot transforms (sampled)
    for k, dec in try_xor_small_keys(payload):
        for m in PRINT_RE.findall(dec):
            candidates.add(m.decode('utf-8', errors='ignore'))
    for dec in try_rot(payload):
        for m in PRINT_RE.findall(dec):
            candidates.add(m.decode('utf-8', errors='ignore'))
    # 4) 滑窗原始二进制用于 n-gram 提取（不在候选串中）
    return list(candidates)

# ----------------- 特征提取: 统计 + k-mer TF-IDF -----------------
def byte_seq_to_str(payload):
    # 将字节序列编码为以空格分隔的 "byte" tokens 供 TfidfVectorizer 使用
    return ' '.join(f'{b:02x}' for b in payload)

def build_feature_table(flow_payloads, tfidf_vectorizer=None, fit_vectorizer=False):
    rows = []
    corpus = []
    for f in flow_payloads:
        p = f['payload'] or b''
        s = byte_seq_to_str(p)
        corpus.append(s)
        rows.append({
            'len': len(p),
            'entropy': byte_entropy(p),
            'printable_ratio': printable_ratio(p),
            'hex_printables_count': len(re.findall(rb'0x[0-9a-fA-F]{2,}', p)),
            'has_ascii': 1 if PRINT_RE.search(p) else 0,
            'payload_str': s,
            'payload_bytes': p,
            'five_tuple': f['five_tuple']
        })
    df = pd.DataFrame(rows)
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'([0-9a-f]{2})', ngram_range=(2,4), max_features=2000)
        X_tfidf = tfidf_vectorizer.fit_transform(df['payload_str'])
    else:
        if fit_vectorizer:
            X_tfidf = tfidf_vectorizer.fit_transform(df['payload_str'])
        else:
            X_tfidf = tfidf_vectorizer.transform(df['payload_str'])
    # 转成稠密小维数组（可改为稀疏传递给模型）
    dense_tfidf = X_tfidf.toarray()
    stat_cols = df[['len','entropy','printable_ratio','hex_printables_count','has_ascii']].values
    X = np.hstack([stat_cols, dense_tfidf])
    return df, X, tfidf_vectorizer

# 统计工具
def byte_entropy(b):
    if not b:
        return 0.0
    counts = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    probs = counts[counts>0] / len(b)
    return -np.sum(probs * np.log2(probs))

def printable_ratio(b):
    if not b:
        return 0.0
    printable = sum(1 for x in b if 32 <= x <= 126)
    return printable / len(b)

# ----------------- 无监督筛选 + 有监督二次确认 -----------------
def unsupervised_candidate_scoring(X, contamination=0.01):
    iso = IsolationForest(contamination=contamination, random_state=42)
    scores = iso.fit_predict(X)  # -1 为异常
    # 这里返回模型和分数数组（负值越小越异常）
    anomaly_scores = iso.decision_function(X)  # 越低越异常
    return iso, anomaly_scores

def train_supervised(X, y):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
    clf.fit(Xs, y)
    return clf, sc

# ----------------- 推理流程整合 -----------------
def infer_pcap(pcap_path, tfidf_vectorizer=None, unsupervised_model=None, supervised_clf=None, scaler=None, top_k=20):
    flows = reassemble_flows(pcap_path)
    if not flows:
        return []
    df, X, tfidf_vectorizer = build_feature_table(flows, tfidf_vectorizer=tfidf_vectorizer, fit_vectorizer=(tfidf_vectorizer is None))
    # 无监督评分
    if unsupervised_model is None:
        unsupervised_model, scores = unsupervised_candidate_scoring(X, contamination=0.02)
    else:
        scores = unsupervised_model.decision_function(X)
    # 取最异常的 N 个流作进一步处理
    idx_sorted = np.argsort(scores)  # 升序，最异常在左
    candidates = []
    top_idx = idx_sorted[:top_k]
    for i in top_idx:
        payload = df.iloc[i]['payload_bytes']
        # 从流尝试抽取候选字符串（解码/滑窗/打印串）
        cand_list = extract_candidates_from_payload(payload)
        # 如果有监督模型可用，用模型评分（模型输入要与训练一致）
        model_score = None
        if supervised_clf is not None and scaler is not None:
            xvec = scaler.transform(X[i].reshape(1, -1))
            model_score = supervised_clf.predict_proba(xvec)[:,1][0]
        candidates.append({
            'five_tuple': df.iloc[i]['five_tuple'],
            'anomaly_score': float(scores[i]),
            'model_score': float(model_score) if model_score is not None else None,
            'candidates': cand_list
        })
    # 汇总并按照 model_score(降序) then anomaly_score(升序) 排序
    candidates = sorted(candidates, key=lambda x: ((-x['model_score']) if x['model_score'] is not None else 0, x['anomaly_score']))
    return candidates, tfidf_vectorizer, unsupervised_model

# ----------------- 示例：训练（假定有带标注的流级样本） -----------------
if __name__ == "__main__":
    # 假设你已经有 labeled_flows: list of {'payload':bytes,'label':0/1}
    # 下面只是示例骨架，数据加载请按你文件组织改写
    labeled_dir_pos = 'labeled/flows/pos'  # 每个文件为 pcap 或用你已有的流序列
    labeled_dir_neg = 'labeled/flows/neg'
    labeled_flows = []
    for d, lab in [(labeled_dir_pos,1),(labeled_dir_neg,0)]:
        if not os.path.exists(d): continue
        for fn in os.listdir(d):
            if not fn.endswith('.pcap'): continue
            flows = reassemble_flows(os.path.join(d, fn))
            for f in flows:
                labeled_flows.append({'payload': f['payload'], 'label': lab})
    if labeled_flows:
        flow_payloads = [{'five_tuple':None,'payload':f['payload']} for f in labeled_flows]
        df, X, tfidf = build_feature_table(flow_payloads)
        y = np.array([f['label'] for f in labeled_flows])
        clf, sc = train_supervised(X, y)
        iso, _ = unsupervised_candidate_scoring(X, contamination=0.02)
        joblib.dump((tfidf, iso, clf, sc), 'flag_pipeline.joblib')
        print("训练完毕并保存: flag_pipeline.joblib")
    else:
        print("未找到标注样本，跳过训练示例。")

    # 推理示例
    if os.path.exists('flag_pipeline.joblib'):
        tfidf, iso, clf, sc = joblib.load('flag_pipeline.joblib')
    else:
        tfidf = iso = clf = sc = None
    pcap_test = 'unknown_capture.pcap'
    if os.path.exists(pcap_test):
        out, tfidf, iso = infer_pcap(pcap_test, tfidf_vectorizer=tfidf, unsupervised_model=iso, supervised_clf=clf, scaler=sc, top_k=50)
        for item in out[:20]:
            print(item)
    else:
        print("测试 pcap 不存在。")
