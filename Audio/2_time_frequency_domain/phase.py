import numpy as np
import matplotlib.pyplot as plt

# 生成一个示例信号（两个正弦波的组合）
t = np.linspace(0, 1, 1000, endpoint=False)
freq1 = 5  # 5 Hz
freq2 = 20  # 20 Hz
signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

# 计算傅里叶变换
fft_result = np.fft.fft(signal)

# 计算频率轴
freqs = np.fft.fftfreq(len(signal), d=t[1] - t[0])

# 提取幅度和相位信息
magnitude = np.abs(fft_result)
phase = np.angle(fft_result)

# 绘制原始信号
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# 绘制幅度谱
plt.subplot(3, 1, 2)
plt.stem(freqs[:len(freqs)//2], magnitude[:len(freqs)//2], 'b', markerfmt=" ", basefmt="-b")
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# 绘制相位谱
plt.subplot(3, 1, 3)
plt.stem(freqs[:len(freqs)//2], phase[:len(phase)//2], 'r', markerfmt=" ", basefmt="-r")
plt.title('Phase Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')

plt.tight_layout()
plt.show()