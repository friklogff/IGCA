import cv2
import numpy as np
import argparse
import functools
from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments
import pyaudio
import librosa

# 音频提取和预处理函数
def extract_audio_from_stream(stream, chunk_size, sample_rate, min_duration=0.4):
    audio_buffer = []
    while True:
        audio_data = np.frombuffer(stream.read(chunk_size), dtype=np.float32)
        audio_buffer.append(audio_data)
        total_duration = len(audio_buffer) * (chunk_size / sample_rate)
        if total_duration >= min_duration:
            break
    audio_data = np.concatenate(audio_buffer)
    return audio_data, sample_rate

def preprocess_audio(audio_data, sample_rate, target_sr=16000):
    if sample_rate != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
    return audio_data

# 设置参数解析
parser = argparse.ArgumentParser(description='实时视频声音识别')
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/cam++.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                  '是否使用GPU预测')
add_arg('model_path',       str,    'models/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=args.use_gpu)

# 打开视频捕获
cap = cv2.VideoCapture(0)  # 使用摄像头

# 设置音频流参数
chunk_size = 1024  # 每次读取的音频帧大小
sample_rate = 44100  # 音频采样率
audio_format = pyaudio.paFloat32  # 音频格式
channels = 1  # 单声道

# 初始化PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=audio_format,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 从音频流中提取音频数据
    audio_data, sample_rate = extract_audio_from_stream(stream, chunk_size, sample_rate)

    # 预处理音频数据
    audio_data = preprocess_audio(audio_data, sample_rate)

    # 实时预测
    probabilities = predictor.mypredict(audio_data=audio_data)

    # 展示预测结果
    predicted_label = max(probabilities, key=probabilities.get)
    predicted_score = probabilities[predicted_label]
    cv2.putText(frame, f"Predicted Sound: {predicted_label} ({predicted_score:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示视频帧
    cv2.imshow('Video Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()
