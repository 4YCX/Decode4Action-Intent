import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from dataset import EEGDataset
from model import CNNLSTMClassifier

# 参数配置
batch_size = 32

# 加载数据集（和train.py保持一致）
root_dir = r"DDG"  
dataset = EEGDataset(root_dir)

# 划分训练集与验证集（同train.py）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTMClassifier(num_channels=2, num_classes=16).to(device)
model.load_state_dict(torch.load("cnn_lstm_model.pth"))
model.eval()

# 验证循环
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"验证集准确率: {correct / total:.4f}")
