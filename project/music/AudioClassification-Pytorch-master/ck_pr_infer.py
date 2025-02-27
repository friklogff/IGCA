import argparse
import functools

import librosa
import numpy as np
from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments

# 设置命令行参数
parser = argparse.ArgumentParser(description='音频分类预测')
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/cam++.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                  '是否使用GPU预测')
add_arg('audio_path',       str,    r'D:\ultralytics-main\zzzaaa_project\music\AudioClassification-Pytorch-master\combined_audio.wav', '音频路径')
add_arg('model_path',       str,    'models/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
add_arg('window_size',      float,  0.5,                   '窗口大小（秒）')
add_arg('hop_length',       float,  0.25,                  '步长（秒）')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=args.use_gpu)

# 读取音频文件
y, sr = librosa.load(args.audio_path, sr=16000)

# 设置窗口大小和步长
window_size_samples = int(args.window_size * sr)
hop_length_samples = int(args.hop_length * sr)

# 划分窗口
windows = [y[i:i + window_size_samples] for i in range(0, len(y) - window_size_samples + 1, hop_length_samples)]

# 对每个窗口进行推理
results = []
for i, window in enumerate(windows):
    # 检查窗口的持续时间
    window_duration = len(window) / sr
    if window_duration < 0.4:
        print(f"窗口 {i+1} 的持续时间 {window_duration:.3f} 秒，小于最小持续时间 0.4 秒，跳过该窗口")
        continue

    # 调用模型进行分类
    probabilities = predictor.mypredict(audio_data=window)

    # 记录结果
    results.append((i, probabilities))

# 打印每个窗口的分类结果
for i, probabilities in results:
    print(f'窗口 {i+1} ({i * args.hop_length} - {(i + 1) * args.hop_length + args.window_size} 秒) 的预测结果概率分布：')
    for label, score in probabilities.items():
        print(f'{label}: {score}')
    print()