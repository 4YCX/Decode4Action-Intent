
import numpy as np
from scipy.io import loadmat
import pywt

def wavelet_denoise(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-level]))
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_denoised = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_denoised, wavelet)[:len(signal)]

def load_emg_dataset(mat_path, window_size=40, step=20, wavelet_denoise_enable=True):
    mat = loadmat(mat_path)
    emg = mat['emg']  # shape [N, 10]
    labels = mat['restimulus'].squeeze()

    if wavelet_denoise_enable:
        for ch in range(emg.shape[1]):
            emg[:, ch] = wavelet_denoise(emg[:, ch])

    X, y = [], []
    for i in range(0, len(emg) - window_size, step):
        label = np.bincount(labels[i:i+window_size]).argmax()
        if label != 0:
            X.append(emg[i:i+window_size])
            y.append(label - 1)

    return np.array(X), np.array(y)
