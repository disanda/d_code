import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# 载入音频文件
y, sr = librosa.load(librosa.example('trumpet'))
print(y.shape)
print(sr)

# 计算STFT
n_fft = 2048
hop_length = 512
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length) # (1025,230)
print(D.shape)

S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) # 计算幅值谱
print(S_db.shape)

# 总时长
total_duration = len(y) / sr

# 时间步长
time_steps = np.arange(D.shape[1]) * hop_length / sr

# 打印总时长和时间步长
print(f'Total Duration: {total_duration:.2f} seconds')
print(f'Time Step: {time_steps[1]:.4f} seconds')

# 可视化幅值谱
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()


# # ---------------------计算幅值谱---------------------
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# # 获取频率轴
# frequencies = np.linspace(0, sr / 2, n_fft // 2 + 1)

# # 打印最大频率
# print(f'Maximum frequency: {frequencies[-1]} Hz')

# # 可视化幅值谱
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram (dB)')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.show()


# # ---------------------线性频率刻度---------------------

# import numpy as np
# import matplotlib.pyplot as plt
# import librosa
# import librosa.display

# # 载入音频文件
# y, sr = librosa.load(librosa.example('trumpet'))

# # 计算STFT
# n_fft = 2048
# hop_length = 512
# D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# # 可视化频谱图
# plt.figure(figsize=(12, 8))

# # 线性频率刻度
# plt.subplot(3, 1, 1)
# librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Linear Frequency')

# # 对数频率刻度
# plt.subplot(3, 1, 2)
# librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Log Frequency')

# # 自定义对数频率刻度（手动）
# plt.subplot(3, 1, 3)
# frequencies = np.linspace(0, sr / 2, n_fft // 2 + 1)
# log_frequencies = np.log10(frequencies + 1e-6)
# plt.imshow(S_db, aspect='auto', origin='lower', extent=[0, S_db.shape[1], log_frequencies[0], log_frequencies[-1]])
# plt.colorbar(format='%+2.0f dB')
# plt.yscale('log')
# plt.title('Log Frequency (manual)')

# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # 原始线性刻度的范围
# linear_ticks = [0, 64, 128, 256, 512, 1024]

# # 计算对数刻度的值
# log_ticks = np.log2(linear_ticks)  # 使用对数底为2的对数计算

# # 绘制示意图
# plt.figure(figsize=(8, 4))
# plt.plot(linear_ticks, np.zeros_like(linear_ticks), 'o')

# # 设置对数刻度的标签和位置
# plt.xticks(linear_ticks, [f'{t:.0f}' for t in linear_ticks])
# plt.gca().set_xscale('log', base=2)
# plt.gca().set_xlim(1, 1024)  # 设置x轴的范围

# plt.xlabel('Original Linear Scale')
# plt.ylabel('Logarithmic Scale')

# plt.grid(True)
# plt.tight_layout()
# plt.show()