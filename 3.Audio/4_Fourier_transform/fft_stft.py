import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# 生成一个示例信号
t = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)

# 将信号转换为 PyTorch 张量
signal_tensor = torch.tensor(signal, dtype=torch.float32)

# 计算 STFT
n_fft = 256
win_length = n_fft
hop_length = win_length // 4
stft_result = torch.stft(signal_tensor, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)

# 计算幅度谱
magnitude = torch.abs(stft_result).numpy()

# 使用 scipy.fftpack.fft 计算 FFT
fft_result = fft(signal)

# 计算频率轴
freqs = np.fft.fftfreq(len(signal), d=t[1] - t[0])

# 可视化 STFT 结果
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.imshow(magnitude, aspect='auto', origin='lower', extent=[0, 1, 0, 0.5 * 1000 / 2])
plt.colorbar()
plt.title('Short-Time Fourier Transform (STFT)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')

# 可视化 FFT 结果
plt.subplot(2, 1, 2)
plt.plot(freqs[:len(freqs)//2], np.abs(fft_result)[:len(fft_result)//2])
plt.title('Fourier Transform (FFT)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()