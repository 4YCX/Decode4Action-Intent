import torch
from torch.utils.data import  DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from dataset import NinaproDataset
from EarlyStop import EarlyStopping
from train import train
from evaluate import evaluate
from preprocess import loadwindow
from model import CNN_TransEncoder


### file path
MAT_FILE = 'D:\DB1_s16\S16_A1_E1.mat'  # 替换为实际路径

### 数据处理
WINDOW_SIZE = 60
STEP_SIZE = 20

### 训练超参
BATCH_SIZE = 128
EPOCHS = 1500
LR = 1e-2

### 训练测试划分
TEST_RATIO = 0.2
RANDOM_STATE = 42

###设备使用
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

### model 参数说明
NUM_CLASSES = 12
CHANNELS = 12
WAVELET_DIM = 120
ENCODER_HEAD = 4

def main():

    # 使用训练器
    print(f"using device {DEVICE}")
    
    # 全新数据加载/划分
    X_emg, X_wave, y = loadwindow(MAT_FILE, window_size=WINDOW_SIZE,
                                   step_size=STEP_SIZE)
    
    X_train, X_val, X_wave_train, X_wave_val, y_train, y_val = train_test_split(
        X_emg, X_wave, y, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y
    )
    # model初始化
    model = CNN_TransEncoder(
        in_channels=X_emg.shape[2],
        wavelet_dim=X_wave.shape[1],
        num_classes=NUM_CLASSES,
        transformer_heads=ENCODER_HEAD
    ).to(DEVICE)

    writer = SummaryWriter()

    # dataset 初始化
    train_loader = DataLoader(NinaproDataset(X_train, X_wave_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(NinaproDataset(X_val, X_wave_val, y_val), batch_size=BATCH_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=20)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=20)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        # 调度
        scheduler.step(val_acc)

        print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if early_stopper(val_loss):
            print(" Early stopping triggered.")
            break

    writer.close()
    torch.save(model.state_dict(), './MODEL/DB1_ExerciseA_CNN_Transformer.pth')
    print("model saved to MODEL")



if __name__ == "__main__":
    main()