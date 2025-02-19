import argparse
import functools

from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description='音频分类预测')
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/cam++.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                  '是否使用GPU预测')
add_arg('audio_path',       str,    'dataset/UrbanSound8K/audio/fold5/156634-5-2-5.wav', '音频路径')
add_arg('model_path',       str,    'models/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=args.use_gpu)

# 调用 my_predict 方法
probabilities = predictor.mypredict(audio_data=args.audio_path)

# 打印每个类别的概率
print(f'音频：{args.audio_path} 的预测结果概率分布：')
for label, score in probabilities.items():
    print(f'{label}: {score}')