import numpy as np
from scipy.io import loadmat

from Advance import wavelet_denoise
import pywt

def extract(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    return [np.sum(np.square(c)) for c in coeffs]  # 每层系数能量




def loadwindow(mat_file, window_size, step_size, wavelet='db4', level=3):
    """
    读取DB1的.mat文件,滑动窗口划分样本并提取频域特征
    
    返回：
    - X_emg: shape (N, T, 10)
    - X_wave: shape (N, 10*(level+1))  # 每通道小波能量
    - y: shape (N,)
    """
    
    mat = loadmat(mat_file)
    emg = mat['emg']          # (N, 10)
    labels = mat['restimulus'].squeeze()  # (N,)

    # 小波去噪
    for ch in range(emg.shape[1]):
        emg[:, ch] = wavelet_denoise(emg[:, ch], wavelet, level)

    X, X_wave, y = [], [], []
    for i in range(0, len(emg) - window_size, step_size):
        window = emg[i:i+window_size]
        label = np.bincount(labels[i:i+window_size]).argmax()
        if label != 0:
            X.append(window)
            y.append(label - 1)

            # 小波能量提取（每通道一组）
            wavelet_feature = []
            for ch in range(window.shape[1]):
                ch_energy = extract(window[:, ch], wavelet, level)
                wavelet_feature.extend(ch_energy)
            X_wave.append(wavelet_feature)

    # 在模型中加入了波形能量的特征通道
    return np.array(X), np.array(X_wave), np.array(y)


