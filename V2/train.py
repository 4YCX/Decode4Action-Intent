import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm
from dataset import EEGDataset
from model import CNNLSTMClassifier

# 参数配置
batch_size = 32
epochs = 20
learning_rate = 0.001

# 加载数据集
root_dir = r"C:\Users\肖长屹Charlie\Desktop\EEG_Transform_DB"  
dataset = EEGDataset(root_dir)

# 划分训练集与验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTMClassifier(num_channels=2, num_classes=16).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")

# 保存模型
torch.save(model.state_dict(), "cnn_lstm_model.pth")

print("训练完成，模型已保存。")
