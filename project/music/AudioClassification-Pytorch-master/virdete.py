import argparse
import functools
import numpy as np
import cv2
import librosa
import moviepy.editor as mp
from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments

# 音频预处理函数
def preprocess_audio(audio_data, sample_rate, target_sr=16000):
    if sample_rate != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
    return audio_data

# 视频帧处理函数
def process_video_frame(frame, predictions):
    y_offset = 30
    for label, score in predictions.items():
        cv2.putText(frame, f"{label}: {score:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_offset += 30
    return frame

# 设置参数解析
parser = argparse.ArgumentParser(description='音频分类预测')
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/cam++.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                  '是否使用GPU预测')
add_arg('model_path',       str,    'models/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
add_arg('audio_path',       str,    'output/combined_audio.wav', '音频路径')
add_arg('output_video',     str,    'output/video_with_annotations.mp4', '输出视频文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=args.use_gpu)

# 加载音频数据
audio_data, sample_rate = librosa.load(args.audio_path, sr=None)

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
    predictions.append((i * segment_duration, probabilities))

# 创建一个空白视频
frame_width, frame_height = 640, 480  # 视频帧的宽度和高度
fps = 30  # 视频帧率
video_duration = num_segments * segment_duration  # 视频总时长

# 生成视频帧
frames = []
for i in range(int(video_duration * fps)):
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    current_time = i / fps
    for start_time, probabilities in predictions:
        if start_time <= current_time < start_time + segment_duration:
            frame = process_video_frame(frame, probabilities)
            break
    frames.append(frame)

# 使用moviepy生成新的视频文件
clip = mp.ImageSequenceClip(frames, fps=fps)
clip.write_videofile("temp_video.mp4", codec='libx264')

# 将音频添加到视频中
video = mp.VideoFileClip("temp_video.mp4")
audio = mp.AudioFileClip(args.audio_path)
final_video = video.set_audio(audio)
final_video.write_videofile(args.output_video, codec='libx264')

print(f'视频生成完成，保存为：{args.output_video}')
