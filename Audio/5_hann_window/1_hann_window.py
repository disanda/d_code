import torch
import numpy as np
import matplotlib.pyplot as plt


###################---case 1---################### 生成汉宁窗口
# # 窗口长度
# N = 50

# # 生成汉宁窗口
# hann_window = np.hanning(N)

# # 绘制汉宁窗口
# plt.plot(hann_window)
# plt.title("Hann Window")
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")
# plt.show()


###################---case 2---################### 填充

# signal = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]) # 模拟信号

# # STFT 参数
# n_fft = 8
# hop_size = 2

# # 对信号进行填充
# padded_signal = torch.nn.functional.pad(signal.unsqueeze(0), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
# padded_signal = padded_signal.squeeze(0)

# print("原始信号长度:", len(signal)) # 5
# print("填充后的信号长度:", len(padded_signal)) # 11
# print(padded_signal) 
# tensor([0.4000, 0.3000, 0.2000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.4000, 0.3000, 0.2000])

###################---case 2---################### 汉宁窗应用

# STFT 参数
n_fft = 2048
hop_size = 512
win_size = 2048 # 窗口长度

# 模拟音频信号
sampling_rate = 16000
t = np.linspace(0, 1, sampling_rate)
freq = 440  # 频率为440Hz的正弦波
y = torch.tensor(np.sin(2 * np.pi * freq * t)).float()

# 生成汉宁窗口
hann_window = torch.hann_window(win_size)

# 对信号进行填充
y = torch.nn.functional.pad(y.unsqueeze(0), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect') # [17536]
y = y.squeeze(0)
print(y.shape)

# ------------------计算 STFT-1
spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, 
    window=hann_window, center=True, pad_mode='reflect', normalized=False, 
    onesided=True, return_complex=True)
print(spec.shape)  # torch.Size([1025, 35]), 查看STFT结果的形状

# 转换为频谱幅度
#real = spec.real # 实部
#imag = spec.imag # 虚部
#spec_magnitude = torch.sqrt(real**2 + imag**2).numpy()
spec_magnitude = spec.abs()
print(spec)

#绘制频谱图
plt.figure(figsize=(10, 6))
plt.imshow(20 * np.log10(spec_magnitude + 1e-6), aspect='auto', origin='lower', extent=[0, t[-1], 0, sampling_rate / 2])
plt.colorbar(format='%+2.0f dB')
plt.title("STFT Magnitude Spectrogram")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.show()

# --------------------计算 STFT-2 without Window or Padding
# spec = torch.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=win_size, window=None, return_complex=True)

# 计算幅度谱
# spec_magnitude = spec.abs()
# real = spec.real # 实部
# imag = spec.imag # 虚部
# spec_magnitude = torch.sqrt(real**2 + imag**2).numpy()

# 绘制频谱图
# plt.figure(figsize=(10, 6))
# plt.imshow(np.log(spec_magnitude + 1e-6), origin='lower', aspect='auto', cmap='inferno', extent=[0, t[-1], 0, sampling_rate / 2])
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram without Window Function')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.show()