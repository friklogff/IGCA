import argparse
import functools
import random

from macls.utils.utils import add_arguments
from pydub import AudioSegment

# 音频合成函数
def concatenate_audio_files(file_paths):
    combined = AudioSegment.empty()
    for file_path in file_paths:
        audio = AudioSegment.from_wav(file_path)
        combined += audio
    return combined

# 设置参数解析
parser = argparse.ArgumentParser(description='音频合成')
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('test_list',        str,    'dataset/test_list.txt', '测试数据集文件路径')
add_arg('output_audio',     str,    'output/combined_audio.wav', '输出音频文件路径')
args = parser.parse_args()

# 读取测试数据集文件
with open(args.test_list, 'r') as file:
    lines = file.readlines()

# 随机选择30个音频文件路径
random.shuffle(lines)
selected_lines = lines[:30]

# 提取音频文件路径
audio_file_paths = [line.strip().split('\t')[0] for line in selected_lines]

# 合成音频
combined_audio = concatenate_audio_files(audio_file_paths)
combined_audio.export(args.output_audio, format="wav")

print(f'音频合成完成，保存为：{args.output_audio}')
