import cv2
import numpy as np
import argparse
import functools
from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments
import librosa
import moviepy.editor as mp
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# 音频提取和预处理函数
def extract_audio_from_video(video_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile("temp_audio.wav")
    audio_data, sample_rate = librosa.load("temp_audio.wav", sr=None)
    return audio_data, sample_rate

def preprocess_audio(audio_data, sample_rate, target_sr=16000):
    if sample_rate != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
    return audio_data

# 视频帧处理函数
def process_video_frame(frame, predicted_label, predicted_score):
    cv2.putText(frame, f"Predicted Sound: {predicted_label} ({predicted_score:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# 设置参数解析
parser = argparse.ArgumentParser(description='视频和音频检测')
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/cam++.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                  '是否使用GPU预测')
add_arg('video_path',       str,    r'D:\ultralytics-main\zzzaaa_project\music\AudioClassification-Pytorch-master\shanghai.mp4', '视频文件路径')
add_arg('output_path',      str,    'output/video.mp4', '输出视频文件路径')
add_arg('model_path',       str,    'models/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=args.use_gpu)

# 提取音频数据
audio_data, sample_rate = extract_audio_from_video(args.video_path)

# 预处理音频数据
audio_data = preprocess_audio(audio_data, sample_rate)

# 分割音频数据并进行预测
segment_duration = 1  # 每段音频的持续时间（秒）
segment_length = int(segment_duration * sample_rate)
num_segments = len(audio_data) // segment_length

predictions = []
for i in range(num_segments):
    segment = audio_data[i * segment_length:(i + 1) * segment_length]
    probabilities = predictor.mypredict(audio_data=segment)
    predicted_label = max(probabilities, key=probabilities.get)
    predicted_score = probabilities[predicted_label]
    predictions.append((i * segment_duration, predicted_label, predicted_score))

# 打开视频文件
cap = cv2.VideoCapture(args.video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 存储处理后的帧
processed_frames = []

current_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 处理视频帧
    for start_time, label, score in predictions:
        if start_time <= current_time < start_time + segment_duration:
            frame = process_video_frame(frame, label, score)
            break

    # 将处理后的帧添加到列表中
    processed_frames.append(frame)

    # 更新当前时间
    current_time += 1 / fps

# 释放资源
cap.release()

# 使用moviepy生成新的视频文件
clip = ImageSequenceClip(processed_frames, fps=fps)
clip.write_videofile(args.output_path, codec='libx264')
