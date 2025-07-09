import torch.nn as nn
import pywt
import numpy as np
### 模型性能强化板块

# 强化指标1: 区别于传统CNN，并非平均CONV和POOL，而是采用前向计算更新通道加权系数
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, C, T]
        y = self.avg_pool(x).squeeze(-1)     # [B, C]
        y = self.fc(y).unsqueeze(-1)         # [B, C, 1]
        return x * y                         # 通道加权
    
# 强化指标2：小波去噪。现在是10通道同样的去噪方式，后续修改思路为10通道分开去噪

def wavelet_denoise(signal, wavelet='db4', level=3):
    """
    对单通道信号进行小波去噪。
    signal: 1D numpy array，表示一个通道的时间序列
    wavelet: 小波类型，例如 'db4'
    level: 小波分解层数
    返回：去噪后的信号
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-level]))
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_denoised = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_denoised, wavelet)[:len(signal)]


# 强化指标3：频域能量特征解码 适配于时空transformer 现在只是简单的利用了特征通道，后续可以转化为时空Transformer

def extract_wavelet_energy(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    return [np.sum(np.square(c)) for c in coeffs]  # 每层系数能量
