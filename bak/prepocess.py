import pandas as pd
import numpy as np
import pyshark
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json


class PCAPPreprocessor:
    def __init__(self):
        self.features = []
        self.labels = []
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def extract_features(self, pcap_file, label=None):
        """从PCAP文件中提取特征"""
        try:
            cap = pyshark.FileCapture(pcap_file, use_json=True, include_raw=False)

            packet_count = 0
            total_size = 0
            protocols = {}
            durations = []
            prev_time = None

            for packet in cap:
                packet_count += 1
                total_size += int(packet.length) if hasattr(packet, 'length') else 0

                # 记录协议类型
                protocol = packet.highest_layer if hasattr(packet, 'highest_layer') else 'Unknown'
                protocols[protocol] = protocols.get(protocol, 0) + 1

                # 计算时间间隔
                if hasattr(packet, 'sniff_time'):
                    if prev_time is not None:
                        durations.append(float(packet.sniff_time.timestamp() - prev_time))
                    prev_time = packet.sniff_time.timestamp()

            cap.close()

            # 计算统计特征
            if durations:
                avg_duration = np.mean(durations)
                std_duration = np.std(durations)
            else:
                avg_duration = std_duration = 0

            # 构建特征向量
            feature_vector = [
                packet_count,  # 包数量
                total_size,  # 总流量大小
                total_size / packet_count if packet_count > 0 else 0,  # 平均包大小
                avg_duration,  # 平均时间间隔
                std_duration,  # 时间间隔标准差
                protocols.get('TCP', 0),  # TCP包数量
                protocols.get('UDP', 0),  # UDP包数量
                protocols.get('ICMP', 0),  # ICMP包数量
            ]

            self.features.append(feature_vector)
            if label is not None:
                self.labels.append(label)

            return feature_vector

        except Exception as e:
            print(f"处理文件 {pcap_file} 时出错: {e}")
            return None

    def process_directory(self, directory, label=None):
        """处理目录中的所有PCAP文件"""
        for filename in os.listdir(directory):
            if filename.endswith('.pcap') or filename.endswith('.pcapng'):
                filepath = os.path.join(directory, filename)
                print(f"处理文件: {filepath}")
                self.extract_features(filepath, label)

    def get_features_and_labels(self):
        """返回特征和标签"""
        features = np.array(self.features)
        labels = np.array(self.labels) if self.labels else None
        return features, labels

    def save_features(self, output_file):
        """保存特征到文件"""
        data = {
            'features': self.features,
            'labels': self.labels if self.labels else []
        }
        with open(output_file, 'w') as f:
            json.dump(data, f)

    def load_features(self, input_file):
        """从文件加载特征"""
        with open(input_file, 'r') as f:
            data = json.load(f)
        self.features = data['features']
        self.labels = data['labels']


# 使用示例
if __name__ == "__main__":
    preprocessor = PCAPPreprocessor()

    # 处理正常流量文件
    preprocessor.process_directory('normal_traffic', label=0)

    # 处理异常流量文件
    preprocessor.process_directory('malicious_traffic', label=1)

    # 保存特征
    preprocessor.save_features('traffic_features.json')
