import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np

class EEGDataset(Dataset):
    def __init__(self, root_dir, max_length=2000):
        self.samples = []
        self.labels = []
        self.label_map = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'wj':10,'wq':11,'ok':12,'right':13,'down':14,'left':15}
        for label_name in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            label_idx = self.label_map[label_name]
            for filename in os.listdir(label_dir):
                if filename.endswith('.csv'):
                    self.samples.append(os.path.join(label_dir, filename))
                    self.labels.append(label_idx)
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        data = pd.read_csv(file_path)
        data = data[['channel_1','channel_2']].values.T
        if data.shape[1] >= self.max_length:
            data = data[:,:self.max_length]
        else:
            pad_width = self.max_length - data.shape[1]
            data = np.pad(data, ((0,0),(0,pad_width)), mode='constant')
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label
