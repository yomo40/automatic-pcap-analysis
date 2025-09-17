import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


class TrafficClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_data(self, features_file):
        """加载特征数据"""
        with open(features_file, 'r') as f:
            data = json.load(f)

        X = np.array(data['features'])
        y = np.array(data['labels'])

        return X, y

    def preprocess_data(self, X, y):
        """预处理数据"""
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 编码标签（如果有必要）
        if y.dtype == np.object or y.dtype.type == np.str_:
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y

        return X_scaled, y_encoded

    def train_models(self, X, y, test_size=0.2):
        """训练多种模型并比较性能"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }

        results = {}
        for name, model in models.items():
            print(f"训练 {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'report': classification_report(y_test, y_pred)
            }
            print(f"{name} 准确率: {accuracy:.4f}")

        return results, X_test, y_test

    def train_neural_network(self, X, y, test_size=0.2):
        """训练神经网络模型"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 构建神经网络
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # 训练模型
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )

        # 评估模型
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"神经网络测试准确率: {test_acc:.4f}")

        return model, history, test_acc

    def save_model(self, model, filename):
        """保存模型"""
        if isinstance(model, keras.Model):
            model.save(filename)
        else:
            joblib.dump(model, filename)
        print(f"模型已保存到 {filename}")

    def load_model(self, filename):
        """加载模型"""
        if filename.endswith('.h5'):
            return keras.models.load_model(filename)
        else:
            return joblib.load(filename)


# 使用示例
if __name__ == "__main__":
    classifier = TrafficClassifier()

    # 加载数据
    X, y = classifier.load_data('RealCheckIn.pcap')

    # 预处理数据
    X_processed, y_processed = classifier.preprocess_data(X, y)

    # 训练传统机器学习模型
    results, X_test, y_test = classifier.train_models(X_processed, y_processed)

    # 训练神经网络
    nn_model, history, nn_accuracy = classifier.train_neural_network(X_processed, y_processed)

    # 保存最佳模型
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']

    if nn_accuracy > results[best_model_name]['accuracy']:
        classifier.save_model(nn_model, 'best_model_nn.h5')
        print("保存神经网络模型")
    else:
        classifier.save_model(best_model, 'best_model_rf.pkl')
        print(f"保存 {best_model_name} 模型")