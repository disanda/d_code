#Fourier transform

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# 生成示例音频信号
sample_rate = 44100  # 采样率
duration = 1.0  # 持续时间
frequency = 440  # 频率 (A4 音符)
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
signal = 0.5 * np.sin(2 * np.pi * frequency * t)

# 计算傅里叶变换
N = len(signal)
yf = fft(signal) # (44100)
xf = np.linspace(0.0, sample_rate / 2.0, N // 2)

print(yf.shape)
print(yf[1] == yf[-1])
print(yf[1].real == yf[-1].real)
print(yf[1].imag == yf[-1].imag*(-1))


# # 可视化时域信号
# plt.figure(figsize=(12, 6))

# plt.subplot(2, 1, 1)
# plt.plot(t, signal)
# plt.title("Time Domain Signal")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")

# # 可视化频域信号
# plt.subplot(2, 1, 2)
# plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
# plt.title("Frequency Domain Signal")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnitude")

# plt.tight_layout()
# plt.show()