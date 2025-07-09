import torch
import torch.nn as nn

class CNNLSTMClassifier(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(CNNLSTMClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.permute(0,2,1)
        out, _ = self.lstm(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return out
