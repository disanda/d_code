import librosa
import librosa.display
import matplotlib.pyplot as plt

# 加载示例音频
audio_path = librosa.example('trumpet')
y, sr = librosa.load(audio_path)
y  = y[0:int(3.5 * sr)] # 取前3.5秒（之后没声音）

# 定义帧大小和帧步长（以秒为单位）
frame_size = 0.025
frame_stride = 0.01

# 将音频分割成短时帧
frames = librosa.util.frame(y, frame_length=int(sr*frame_size), hop_length=int(sr*frame_stride))

# 计算时间轴（以秒为单位）
t = librosa.frames_to_time(range(len(frames[0])), sr=sr, hop_length=int(sr*frame_stride))

# 可视化分割后的帧
plt.figure(figsize=(10, 4))
#librosa.display.waveshow(y, sr=sr, color='g', alpha=0.5)  # 显示完整波形
librosa.display.waveshow(frames, sr=sr, color='r', alpha=0.5)  # 分割后的帧
plt.title('Waveform and Frames')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['Waveform', 'Frames'])
plt.tight_layout()
plt.show()