import numpy as np
import matplotlib.pyplot as plt
import torch

# 生成一个示例信号
sampling_rate = 1000
t = np.linspace(0, 1, sampling_rate, endpoint=False)
freq = 5  # 频率为5Hz的正弦波
signal = np.sin(2 * np.pi * freq * t)

# 定义窗口函数
window_size = 256
hann_window = torch.hann_window(window_size) # $w(n)=0.5(1−cos\frac{N−1}{2πn}), n \in [0, N-1]$

# 应用窗口函数
windowed_signal = signal[:window_size] * hann_window.numpy()

# 绘制信号和窗口函数
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(signal[:window_size], label="Original Signal")
plt.plot(hann_window.numpy(), label="Hanning Window")
plt.legend()
plt.title("Original Signal and Hanning Window")

plt.subplot(2, 1, 2)
plt.plot(windowed_signal, label="Windowed Signal")
plt.legend()
plt.title("Windowed Signal")

plt.tight_layout()
plt.show()