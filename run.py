import pickle

# 加载模型
with open(".pkl", "rb") as f:
    model = pickle.load(f)

# 输入数据示例（需要与你训练时的特征维度一致）
X_test = [[1.2, 3.4, 5.6, 7.8]]

# 推理
y_pred = model.predict(X_test)
print("预测结果:", y_pred)
