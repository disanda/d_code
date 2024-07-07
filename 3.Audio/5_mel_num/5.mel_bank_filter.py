import numpy
import librosa
import matplotlib.pyplot as plt

y, sample_rate = librosa.load(librosa.example('trumpet')) # (117601,22050)
pre_emphasis = 0.97
y_preemphasized = numpy.append(y[0], y[1:] - pre_emphasis * y[:-1]) 

#-------------------------- slide frame ----------------------

frame_size = 0.025
frame_stride = 0.01 # 10ms stide 帧移时间，非重叠部分。 意味着frame之间间隔(重叠)有 25- 10 = 15ms

frame_length = frame_size * sample_rate  # seconds to samples
frame_step = frame_stride * sample_rate

signal_length = len(y_preemphasized) # 117601 个采样点
frame_length = int(round(frame_length)) # 每个frame 551个采样点，每个0.025秒
frame_step = int(round(frame_step)) # 帧移采样点部分，非重合部分为220个采样点，即10ms
num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # 533，Make sure that we have at least 1 frame


pad_signal_length = num_frames * frame_step + frame_length # 117260（没有覆盖全部杨本点） + 511 = 117811 
z = numpy.zeros((pad_signal_length - signal_length)) # 117811 - 117601 = 210 覆盖全部样本点需要额外210个点
pad_signal = numpy.append(y_preemphasized, z) # 填充210个点，保证覆盖全部样本点

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
# frames的索引 [0, 550], [220, 770],..., [117040, 117590]
#print(indices[3])
#-------------------------- mel ----------------------

num_mel = 40
low_freq_mel = 0
N = 512 # NFFT

frames = pad_signal[indices.astype(numpy.int32, copy=False)] # (533, 551)
frames *= numpy.hamming(frame_length)
mag_frames = numpy.absolute(numpy.fft.rfft(frames, N))  # Magnitude of the FFT (533, 257)
pow_frames = ((1.0 / N) * ((mag_frames) ** 2))  # Power Spectrum (533, 257)

high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = numpy.linspace(low_freq_mel, high_freq_mel, num_mel + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = numpy.floor((N + 1) * hz_points / sample_rate)

fbank = numpy.zeros((num_mel, int(numpy.floor(N / 2 + 1)))) # (num_mel, N /2 + 1)
for m in range(1, num_mel + 1):   # 依次制作不同频率的三角滤波器
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

filter_banks = numpy.dot(pow_frames, fbank.T) # (533,257) x (257, num_mel) = (533, 257)
filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # 替换0值，方便转db
filter_banks = 20 * numpy.log10(filter_banks)  # 通过log将梅尔组各个三角频率的能量值转换为分贝（dB）

filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)

# Assuming filter_banks is already calculated
plt.figure(figsize=(10, 6))
plt.imshow(filter_banks.T, origin='lower', aspect='auto', cmap='jet', interpolation='nearest')
plt.title('Mel Filter Bank Spectrogram')
plt.ylabel('Mel Filter Bank')
plt.xlabel('Time Frame')
plt.colorbar(format='%+2.0f dB')
plt.show()

# from scipy.fftpack import dct

# # Apply DCT to the filter banks
# num_ceps = 12
# mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps + 1)]

# cep_lifter = 22 # 22-27
# (nframes, ncoeff) = mfcc.shape
# n = numpy.arange(ncoeff)
# lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter) # 正弦提升,以弱化更高的 MFCC，提高噪声信号中的语音识别能力。
# mfcc *= lift  #*

# mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

# # Plot the MFCCs
# plt.figure(figsize=(10, 6))
# plt.imshow(mfcc.T, origin='lower', aspect='auto', cmap='jet', interpolation='nearest')
# plt.title('MFCC')
# plt.ylabel('Cepstral Coefficient')
# plt.xlabel('Time Frame')
# plt.colorbar(format='%+2.0f dB')
# plt.show()