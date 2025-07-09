import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from Advance import ChannelAttention
             # 通道加权

class CNN_TransEncoder(nn.Module):
    def __init__(self, in_channels, wavelet_dim, num_classes, transformer_heads):
        super(CNN_TransEncoder,self).__init__()

        self.channel_att = ChannelAttention(in_channels)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

        )

        # Transformer Encoder 替代 LSTM
        encoder_layer = TransformerEncoderLayer(d_model=128, nhead=transformer_heads)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)

        # 小波分支
        # 模型再次强化了小波解码的特征信息的使用，需要多输入一路信息
        self.wavelet_fc = nn.Sequential(
            nn.Linear(wavelet_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_emg, x_wavelet):
        # x_emg: [B, T, C]
        x_emg = x_emg.permute(0, 2, 1)           # [B, C, T]
        x_emg = self.channel_att(x_emg)          # 通道注意力
        x = self.cnn(x_emg)                      # [B, 128, T']
        x = x.permute(2, 0, 1)                   # [T', B, 128]

        x_trans = self.transformer(x)            # [T', B, 128]
        x_trans = x_trans.mean(dim=0)            # [B, 128] → 时间平均池化

        x_wave = self.wavelet_fc(x_wavelet)      # [B, 128]
        fused = torch.cat([x_trans, x_wave], dim=1)  # [B, 256]
        return self.classifier(fused)            # [B, num_classes]
