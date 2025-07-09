
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from model import LSTMClassifier
from loaddata import load_emg_dataset

# ==== 超参数 ====
MAT_FILE = "D:\DB1_s16\S16_A1_E1.mat"
NUM_CLASSES = 12
BATCH_SIZE = 64
EPOCHS = 500
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 40
STEP = 20

# ==== 数据加载 ====
X, y = load_emg_dataset(MAT_FILE, window_size=WINDOW_SIZE, step=STEP)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.long))
val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                       torch.tensor(y_val, dtype=torch.long))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ==== 模型 ====
model = LSTMClassifier(input_size=10, hidden_size=128, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==== 训练 ====
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            out = model(X_batch)
            pred = out.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

    print(f"Epoch {epoch:02d}: Loss = {total_loss / len(train_ds):.4f}, Val Acc = {correct / total:.4f}")

torch.save(model.state_dict(), "V1/lstm_emg.pth")
print("模型已保存")
