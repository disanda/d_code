#-------------------------------- Mel_filter_bank 1 ------------------------

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# 设置参数
sr = 22050  # 采样率
n_fft = 2048  # FFT点数
n_mels = 10  # 梅尔滤波器数量

# 生成梅尔滤波器组
mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels) # (n_mels, n_fft/2 + 1)
print(mel_filters.shape)

# # 生成频率轴
# fft_freqs = np.linspace(0, sr // 2, n_fft // 2 + 1)

# # 绘制梅尔滤波器组
# plt.figure(figsize=(12, 6))
# for i in range(n_mels):
#     plt.plot(fft_freqs, mel_filters[i])
# plt.title('Mel Filter Bank1')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.grid()
# plt.show()

#-------------------------------- Mel_filter_bank 2 ------------------------

import librosa
import matplotlib.pyplot as plt
import numpy as np

# 参数设置
sr = 22050          # 采样率
n_fft = 2048        # FFT 大小
n_mels = 128        # 梅尔滤波器数量
fmin = 0.0          # 最低频率
fmax = sr / 2.0     # 最高频率，设置为奈奎斯特频率

# 生成梅尔滤波器组
mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

# 绘制梅尔滤波器
plt.figure(figsize=(10, 6))
plt.imshow(mel_basis, aspect='auto', origin='lower')
plt.colorbar()
plt.title('Mel filter bank')
plt.xlabel('Frequency bins')
plt.ylabel('Mel filters')
plt.savefig('./Mel_filter_bank2.png')
plt.show()