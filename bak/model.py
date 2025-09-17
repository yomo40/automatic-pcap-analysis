from scapy.all import rdpcap, TCP, UDP, IP
import base64, zlib, codecs, re, torch, torch.nn as nn

# 读取并聚合流
packets = rdpcap("RealCheckIn.pcap")            # 读取PCAP
# 按五元组分组，每个流为(pkt.src, pkt.sport, pkt.dst, pkt.dport, proto)
streams = {}
for pkt in packets:
    if IP not in pkt:
        continue
    proto = None
    if TCP in pkt:
        proto = 'TCP'
        l4 = pkt[TCP]
    elif UDP in pkt:
        proto = 'UDP'
        l4 = pkt[UDP]
    else:
        continue
    key = (pkt[IP].src, l4.sport, pkt[IP].dst, l4.dport, proto)
    streams.setdefault(key, b"")
    streams[key] += bytes(l4.payload)

# 解码尝试示例
def try_decodings(data):
    """对数据尝试多种解码，返回原始和所有解码结果字典。"""
    results = {'raw': data}
    try:
        results['base64'] = base64.b64decode(data)
    except Exception:
        pass
    try:
        # 仅当data解码为ascii hex字符串时使用
        hex_str = data.decode('ascii')
        results['hex'] = bytes.fromhex(hex_str)
    except Exception:
        pass
    # XOR with 0xFF as example
    results['xor_ff'] = bytes([b ^ 0xFF for b in data])
    try:
        # ROT13 仅对可打印ASCII进行示例
        text = data.decode('ascii', errors='ignore')
        rot13 = codecs.encode(text, 'rot_13').encode('ascii', errors='ignore')
        results['rot13'] = rot13
    except Exception:
        pass
    try:
        results['zlib'] = zlib.decompress(data)
    except Exception:
        pass
    return results

# 构建标签和特征（示例：若某流的任何解码后包含"flag{"视为正样本）
features = []
labels = []
for key, payload in streams.items():
    decs = try_decodings(payload)
    has_flag = False
    for mode, data in decs.items():
        if b"flag{" in data or b"FLAG{" in data:  # 简单判断
            has_flag = True
    # 示例：只用原始raw字节作为输入（可替换为任意特征向量）
    # 这里简单截断/补齐到固定长度，如256字节（可自行调整）
    seq = payload[:256].ljust(256, b'\x00')
    features.append(torch.tensor(list(seq), dtype=torch.long))
    labels.append(1 if has_flag else 0)
features = torch.stack(features)  # (num_streams, 256)
labels = torch.tensor(labels)

# 定义模型：1层嵌入 + 1层Conv1D + 池化 + 全连接
class FlowClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(256, 16)  # 字节到16维向量
        self.conv = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        # x: (batch, seq_len), 嵌入->(batch, seq_len, 16), 转换维度->(batch,16,seq_len)
        x = self.embed(x).permute(0,2,1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)  # (batch,32)
        x = torch.sigmoid(self.fc(x)).squeeze(-1)
        return x

model = FlowClassifier()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型（示例：若样本少可循环多轮）
for epoch in range(100):
    model.train()
    preds = model(features)
    loss = criterion(preds, labels.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 保存模型（可选）
torch.save(model.state_dict(), "flow_model.pth")
