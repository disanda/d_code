# mel谱 <=> 音频

import numpy as np
import librosa
import soundfile as sf  # 导入 soundfile 库

# 加载示例音频
#y, sr = librosa.load(librosa.ex('trumpet')) #yc_en.m4a
y, sr = librosa.load('./audio_data/syz.mp3', sr=None) #mono=False


# 生成梅尔频谱图
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

# 可视化梅尔频谱图
import librosa.display
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

# 逆变换：从梅尔频谱图恢复音频信号
# Step 1: 将梅尔频谱图转换回线性频谱图
mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=128)
mel_to_linear = np.dot(np.linalg.pinv(mel_basis), mel_spectrogram)

# Step 2: 使用 Griffin-Lim 算法从线性频谱图恢复音频
recovered_audio = librosa.feature.inverse.griffinlim(mel_to_linear) 
#由于梅尔频谱图和频谱图通常只包含幅度信息，恢复音频信号时必须估计相位信息。常用的方法如 Griffin-Lim 算法通过迭代优化来逼近真实的相位。

# 使用 soundfile 保存恢复的音频
sf.write('original_audio.wav', y, sr)
sf.write('recovered_audio_magnitude_only.wav', recovered_audio, sr)


#-------------Magnitude + Phase --------------
# 进行短时傅里叶变换 (STFT)
D = librosa.stft(y)

# 提取幅度谱和相位谱
magnitude = np.abs(D)
phase = np.angle(D)

# 重构复数频域信号
reconstructed_D = magnitude * np.exp(1j * phase)

# 进行逆短时傅里叶变换 (ISTFT)
reconstructed_y = librosa.istft(reconstructed_D)

# 保存重构的音频
sf.write('reconstructed_audio_Mag+Phase.wav', reconstructed_y, sr)

#-------- phase_only 假设幅度为1------------
magnitude = np.ones_like(phase)

# 重构复数频域信号
reconstructed_D = magnitude * np.exp(1j * phase)

# 进行逆短时傅里叶变换 (ISTFT)
reconstructed_y = librosa.istft(reconstructed_D)

# 保存重构的音频
sf.write('reconstructed_from_phase_only.wav', reconstructed_y, sr)