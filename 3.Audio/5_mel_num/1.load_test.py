import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


#----------------------------- Hamming ----------------------
import numpy as np
import matplotlib.pyplot as plt

# 生成200个样本的Hamming窗口
hamming_window = np.hamming(200)

# 归一化使得振幅为1
hamming_window /= np.max(hamming_window)

# 绘制Hamming窗口
plt.plot(hamming_window)
plt.title('Hamming Window')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# -------------------------------- Apply pre-emphasis filter --------------------------------
# y, sr = librosa.load(librosa.example('trumpet'))
# y  = y[0:int(3.5 * sr)]
# pre_emphasis = 0.97
# y_preemphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

# # Plot the original waveform
# plt.figure(figsize=(14, 5))
# plt.subplot(2, 1, 1)
# librosa.display.waveshow(y, sr=sr)
# plt.title('Original Waveform')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')

# # Plot the pre-emphasized waveform
# plt.subplot(2, 1, 2)
# librosa.display.waveshow(y_preemphasized, sr=sr)
# plt.title('Pre-Emphasized Waveform')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')

# plt.tight_layout()
# plt.show()