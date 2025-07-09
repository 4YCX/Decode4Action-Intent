import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class NinaproDataset(Dataset):
    def __init__(self, X_emg, X_wave, y):
        self.X_emg = torch.tensor(X_emg, dtype=torch.float32)      # [N, 40, 10]
        self.X_wave = torch.tensor(X_wave, dtype=torch.float32)    # [N, wavelet_dim]
        self.y = torch.tensor(y, dtype=torch.long)                 # [N]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_emg[idx], self.X_wave[idx], self.y[idx]


