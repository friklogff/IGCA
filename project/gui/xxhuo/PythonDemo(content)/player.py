from pydub import AudioSegment
from pydub.playback import play

# 加载PCM文件
audio = AudioSegment.from_file(
    "demo.pcm",
    format="pcm",
    frame_rate=16000,  # 采样率，根据你的PCM文件调整
    channels=1,        # 通道数，1表示单声道，2表示立体声
    sample_width=2     # 采样宽度，2表示16位
)

# 播放音频
play(audio)
