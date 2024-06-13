import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

array, sampling_rate = librosa.load(librosa.ex("trumpet"))

###------------时域图----------------

# plt.figure().set_figwidth(12)
# librosa.display.waveshow(array, sr=sampling_rate)
# # y轴表示的是信号的幅值，x轴则表示时间

# # 保存图像到文件
# plt.savefig("trumpet_waveform.png")  # 保存为 'trumpet_waveform.png'
# plt.show()


###------------频域图----------------
# import numpy as np

# dft_input = array[:4096]

# # 计算 DFT
# window = np.hanning(len(dft_input))
# windowed_input = dft_input * window
# dft = np.fft.rfft(windowed_input)

# # 计算频谱的幅值，转换为分贝标度
# amplitude = np.abs(dft)
# amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)

# # 计算每个DFT分量对应的频率值
# frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))

# plt.figure().set_figwidth(12)
# plt.plot(frequency, amplitude_db)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude (dB)")
# plt.xscale("log")
# plt.savefig("frequency_domain_plot.png")  # 保存为 'trumpet_waveform.png'
# plt.show()

##-------------时频谱------------
# D = librosa.stft(array) # Short Time Fourier Transform, STFT
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# plt.figure().set_figwidth(12)
# librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
# plt.colorbar()
# plt.savefig("spectrogram_plot.png")  # 保存为 'trumpet_waveform.png'
# plt.show()

##-------------时频谱------------
S = librosa.feature.melspectrogram(y=array, sr=sampling_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure().set_figwidth(12)
librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000)
plt.colorbar()
plt.savefig("mel_spectrogram.png")  # 保存为 'trumpet_waveform.png'
plt.show()