import argparse
import functools

import librosa
import numpy as np
from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments

# 设置命令行参数
parser = argparse.ArgumentParser(description='音频分类预测')
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/cam++.yml',   '配置文件路径')
add_arg('use_gpu',          bool,   True,                  '是否使用GPU进行预测')
add_arg('audio_path',       str,    r'D:\ultralytics-main\zzzaaa_project\music\AudioClassification-Pytorch-master\output\combined_audio.wav', '音频文件路径')
add_arg('model_path',       str,    'models/CAMPPlus_Fbank/best_model/', '模型文件路径')
add_arg('window_size',      float,  4.7,                   '窗口大小（秒）')
add_arg('hop_length',       float,  2.7,                   '步长（秒）')
args = parser.parse_args()
print_arguments(args=args)

# 获取音频分类器
predictor = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=args.use_gpu)

# 读取音频文件
y, sr = librosa.load(args.audio_path, sr=16000)  # 使用16000Hz的采样率加载音频

# 将窗口大小和步长从秒转换为样本数
window_size_samples = int(args.window_size * sr)  # 窗口大小（样本数）
hop_length_samples = int(args.hop_length * sr)    # 步长（样本数）

# 划分音频为多个窗口
windows = [y[i:i + window_size_samples] for i in range(0, len(y), hop_length_samples)]

# 对每个窗口进行推理
results = []
for i, window in enumerate(windows):
    # 检查窗口的持续时间是否满足模型的最小持续时间要求（0.4秒）
    window_duration = len(window) / sr
    if window_duration < 0.4:
        print(f"窗口 {i+1} 的持续时间 {window_duration:.3f} 秒，小于最小持续时间 0.4 秒，跳过该窗口")
        continue

    # 调用模型进行分类
    probabilities = predictor.mypredict(audio_data=window)

    # 记录结果
    start_time = i * args.hop_length  # 窗口的起始时间
    end_time = start_time + args.window_size  # 窗口的结束时间
    results.append((start_time, end_time, probabilities))

# 合并连续相同分类的窗口，并处理重叠
events = []
current_event = None

for start_time, end_time, probabilities in results:
    label = max(probabilities, key=probabilities.get)  # 获取最高概率的标签

    if current_event is None:
        # 如果当前没有事件，初始化一个新事件
        current_event = {'start': start_time, 'end': end_time, 'label': label}
    elif label == current_event['label']:
        # 如果当前窗口的标签与当前事件的标签相同，扩展事件的结束时间
        current_event['end'] = max(current_event['end'], end_time)
    else:
        # 如果标签不同，检查是否有重叠
        if start_time < current_event['end']:
            # 如果有重叠，将当前事件分割为两部分
            # 第一部分：当前事件的开始到新窗口的开始
            events.append({'start': current_event['start'], 'end': start_time, 'label': current_event['label']})
            # 第二部分：新窗口的开始到新窗口的结束
            current_event = {'start': start_time, 'end': end_time, 'label': label}
        else:
            # 如果没有重叠，保存当前事件并开始一个新的事件
            events.append(current_event)
            current_event = {'start': start_time, 'end': end_time, 'label': label}

# 添加最后一个事件
if current_event:
    events.append(current_event)
count=0
# 打印事件检测结果
for event in events:
    count+=1
    print(f" {count} 事件：{event['label']}，时间范围：{event['start']:.2f} - {event['end']:.2f} 秒")

