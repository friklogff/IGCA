# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：trian.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""
# 忽略警告信息，以减少控制台输出的干扰
import warnings

warnings.filterwarnings('ignore')

# 从ultralytics库导入YOLO类，用于创建和训练YOLO模型
from ultralytics import YOLO

# 检查是否为主程序运行，如果是，则执行以下代码
if __name__ == '__main__':
    # 创建YOLO模型实例，指定模型配置文件路径
    # 注释掉的model.load('yolo11n.pt')是加载预训练权重的代码，这里不加载是为了从头开始训练
    # model = YOLO(model='yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开
    model = YOLO('./ultralytics/cfg/models/11/yolo11.yaml')
    model.load('./yolo11n.pt')

    # 调用model对象的train方法开始训练模型
    # 传入训练所需的参数
    model.train(
        data=r'data2.yaml',  # 数据集配置文件的路径
        imgsz=640,  # 训练时输入图像的大小
        epochs=10,  # 训练的总轮数
        batch=4,  # 每批处理的图像数量
        workers=0,  # 用于数据加载的工作线程数，0表示使用主线程
        device='',  # 不指定设备，让YOLO自动选择
        optimizer='SGD',  # 使用SGD优化器
        close_mosaic=1,  # 在前10个epoch不使用mosaic数据增强
        resume=False,  # 不从上次训练中断处继续训练
        project='runs/train',  # 训练项目的保存路径
        name='jiedao',  # 训练实验的名称
        single_cls=False,  # 是否为单类别检测
        cache=False  # 是否缓存数据增强效果
    )