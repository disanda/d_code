# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fftpack import fft
# from scipy.signal import spectrogram
# import matplotlib.gridspec as gridspec

# # 生成音频信号
# def generate_signal(freq, duration, sample_rate):
#     t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
#     signal = 0.5 * np.sin(2 * np.pi * freq * t)
#     return t, signal

# # 参数设置
# sample_rate = 44100
# duration = 1.0  # seconds
# time_window = 0.1  # seconds for time domain plot

# # 不同频段的信号
# signals = {
#     'Infrasound': 10,
#     'Low Frequency': 100,
#     'Mid Frequency': 1000,
#     'High Frequency': 10000,
#     'Ultrasound': 25000
# }

# # 初始化绘图
# fig = plt.figure(figsize=(17, 7))
# gs = gridspec.GridSpec(len(signals), 4, width_ratios=[2, 1, 1, 0.1])
# fig.subplots_adjust(hspace=0.5, wspace=0.3)

# # 生成并可视化每个信号
# for i, (title, freq) in enumerate(signals.items()):
#     t, signal = generate_signal(freq, duration, sample_rate)
    
#     # 时域图 - 显示前time_window秒
#     time_index = int(time_window * sample_rate)
#     ax_time = fig.add_subplot(gs[i, 0])
#     ax_time.plot(t[:time_index], signal[:time_index])
#     ax_time.set_title(f'{title} - Time Domain', fontsize=10)
#     ax_time.set_xlabel('Time [s]', fontsize=8)
#     ax_time.set_ylabel('Amplitude', fontsize=8)
    
#     # 频域图
#     N = len(signal)
#     yf = fft(signal)
#     xf = np.linspace(0.0, sample_rate / 2.0, N // 2)
#     ax_freq = fig.add_subplot(gs[i, 1])
#     ax_freq.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
#     ax_freq.set_title(f'{title} - Frequency Domain', fontsize=10)
#     ax_freq.set_xlabel('Frequency [Hz]', fontsize=8)
#     ax_freq.set_ylabel('Magnitude', fontsize=8)
    
#     # 时频谱图
#     f, t_spec, Sxx = spectrogram(signal, sample_rate)
#     ax_spec = fig.add_subplot(gs[i, 2])
#     cax = ax_spec.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
#     ax_spec.set_title(f'{title} - Spectrogram', fontsize=10)
#     ax_spec.set_xlabel('Time [s]', fontsize=8)
#     ax_spec.set_ylabel('Frequency [Hz]', fontsize=8)
    
#     # 添加颜色条
#     cbar = plt.colorbar(cax, ax=ax_spec, aspect=5)
#     cbar.set_label('Intensity [dB]', fontsize=8)

# plt.tight_layout()
# plt.show()

############################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import spectrogram
import matplotlib.gridspec as gridspec

# 生成音频信号并标准化功率
def generate_signal(freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    # 标准化功率
    signal = signal / np.sqrt(np.mean(signal**2))
    return t, signal

# 参数设置
sample_rate = 44100
duration = 1.0  # seconds
time_window = 0.1  # seconds for time domain plot

# 不同频段的信号
signals = {
    'Infrasound': 10,
    'Low Frequency': 100,
    'Mid Frequency': 1000,
    'High Frequency': 10000,
    'Ultrasound': 25000
}

# 初始化绘图
fig = plt.figure(figsize=(17, 7))
gs = gridspec.GridSpec(len(signals), 4, width_ratios=[2, 1, 1, 0.1])
fig.subplots_adjust(hspace=0.5, wspace=0.3)

# 生成并可视化每个信号
for i, (title, freq) in enumerate(signals.items()):
    t, signal = generate_signal(freq, duration, sample_rate)
    
    # 时域图 - 显示前time_window秒
    time_index = int(time_window * sample_rate)
    ax_time = fig.add_subplot(gs[i, 0])
    ax_time.plot(t[:time_index], signal[:time_index])
    ax_time.set_title(f'{title} - Time Domain', fontsize=10)
    ax_time.set_xlabel('Time [s]', fontsize=8)
    ax_time.set_ylabel('Amplitude', fontsize=8)
    
    # 频域图
    N = len(signal)
    yf = fft(signal)
    xf = np.linspace(0.0, sample_rate / 2.0, N // 2)
    ax_freq = fig.add_subplot(gs[i, 1])
    ax_freq.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    ax_freq.set_title(f'{title} - Frequency Domain', fontsize=10)
    ax_freq.set_xlabel('Frequency [Hz]', fontsize=8)
    ax_freq.set_ylabel('Magnitude', fontsize=8)
    
    # 时频谱图
    f, t_spec, Sxx = spectrogram(signal, sample_rate)
    ax_spec = fig.add_subplot(gs[i, 2])
    cax = ax_spec.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    ax_spec.set_title(f'{title} - Spectrogram', fontsize=10)
    ax_spec.set_xlabel('Time [s]', fontsize=8)
    ax_spec.set_ylabel('Frequency [Hz]', fontsize=8)
    
    # 添加颜色条
    cbar = plt.colorbar(cax, ax=ax_spec, aspect=5)
    cbar.set_label('Intensity [dB]', fontsize=8)

plt.tight_layout()
plt.show()
