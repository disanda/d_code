import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import glob

# 设置文件目录和格式
file_path_pattern = './wavs/LJ050-*.wav'  # 替换为实际LJSpeech文件所在的路径

# 收集所有LJ050的音频文件
audio_files = glob.glob(file_path_pattern)

# 只加载前30个音频文件
audio_files = sorted(audio_files)[:30]

# 初始化空的音频序列
full_audio = np.array([])

# 加载和合并前30个音频文件
for file in audio_files:
    audio, sr = librosa.load(file, sr=None)
    full_audio = np.concatenate((full_audio, audio))

# 创建图形窗口
plt.figure(figsize=(31, 23))

# 时域图
plt.subplot(4, 1, 1)
librosa.display.waveshow(full_audio, sr=sr)
plt.title('Time Domain (Waveform)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# 频域图 (FFT)
plt.subplot(4, 1, 2)
D = np.fft.fft(full_audio)
frequencies = np.fft.fftfreq(len(D), 1 / sr)
plt.plot(frequencies[:len(frequencies)//2], np.abs(D[:len(D)//2]))
plt.title('Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# # 相位图 (Phase Plot)
# plt.subplot(4, 1, 3)
# plt.plot(frequencies[:len(frequencies)//2], np.angle(D[:len(D)//2]))
# plt.title('Phase Plot')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Phase (radians)')

# 频谱图 (Spectrogram)
plt.subplot(4, 1, 3)
S = np.abs(librosa.stft(full_audio))
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

# 梅尔谱图 (Mel Spectrogram)
plt.subplot(4, 1, 4)
S_mel = librosa.feature.melspectrogram(y=full_audio, sr=sr, n_mels=128)
librosa.display.specshow(librosa.power_to_db(S_mel, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Mel Frequency')

# 调整布局并显示
# plt.tight_layout()
# plt.show()
plt.savefig("mel_spectrogram_2.png")