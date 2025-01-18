# 导入所需要的库
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np
from PIL import Image

import paddle
import paddle.nn as nn
from paddle.io import Dataset
import paddle.vision.transforms as T
import paddle.nn.functional as F
from paddle.metric import Accuracy

import warnings
warnings.filterwarnings("ignore")

# 数据EDA
df = pd.read_csv('data/train_data/train_label.csv')
d = df['label'].hist().get_figure()
print(d)

# 读取数据
train_images = pd.read_csv('data/train_data/train_label.csv', usecols=['image_name','label'])  # 读取文件名和类别

# labelshuffling，定义标签打乱模块
def labelShuffling(dataFrame, groupByName='label'):
    groupDataFrame = dataFrame.groupby(by=[groupByName])
    labels = groupDataFrame.size()
    print("length of label is ", len(labels))
    maxNum = max(labels)
    lst = pd.DataFrame()
    for i in range(len(labels)):
        print("Processing label  :", i)
        tmpGroupBy = groupDataFrame.get_group(i)
        createdShuffleLabels = np.random.permutation(np.array(range(maxNum))) % labels[i]  # 随机排列组合
        print("Num of the label is : ", labels[i])
        lst = lst.append(tmpGroupBy.iloc[createdShuffleLabels], ignore_index=True)
    return lst

# 进行训练集和测试集划分，按照先打乱后划分原则
all_size = len(train_images)
print("训练集大小：", all_size)
train_size = int(all_size * 0.8)
train_image_list = train_images[:train_size]
val_image_list = train_images[train_size:]

df = labelShuffling(train_image_list)
df = shuffle(df)
print("shuffle后数据集大小：", len(df))

train_image_path_list = df['image_name'].values
label_list = df['label'].values
label_list = paddle.to_tensor(label_list, dtype='int64')
train_label_list = paddle.nn.functional.one_hot(label_list, num_classes=4)

val_image_path_list = val_image_list['image_name'].values
val_label_list = val_image_list['label'].values
val_label_list = paddle.to_tensor(val_label_list, dtype='int64')
val_label_list = paddle.nn.functional.one_hot(val_label_list, num_classes=4)

# 定义数据预处理,数据增广方法
data_transforms = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(3. / 4, 4. / 3), interpolation='bilinear'),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.Transpose(),    # HWC -> CHW
    T.Normalize(
        mean=[127.5, 127.5, 127.5],        # 归一化
        std=[127.5, 127.5, 127.5],
        to_rgb=True)
])

# 构建Dataset
class MyDataset(paddle.io.Dataset):
    def __init__(self, train_img_list, val_img_list, train_label_list, val_label_list, mode='train'):
        super(MyDataset, self).__init__()
        self.img = []
        self.label = []
        self.train_images = train_img_list
        self.val_images = val_img_list
        self.train_label = train_label_list
        self.val_label = val_label_list
        if mode == 'train':
            for img, la in zip(self.train_images, self.train_label):
                self.img.append('data/train_data/train_image/' + img)
                self.label.append(la)
        else:
            for img, la in zip(self.val_images, self.val_label):
                self.img.append('data/train_data/train_image/' + img)
                self.label.append(la)

    def load_img(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

    def __getitem__(self, index):
        image = self.load_img(self.img[index])
        label = self.label[index]
        return data_transforms(image), label

    def __len__(self):
        return len(self.img)

BATCH_SIZE = 128
PLACE = paddle.CUDAPlace(0)

# train_loader
train_dataset = MyDataset(
    train_img_list=train_image_path_list,
    val_img_list=val_image_path_list,
    train_label_list=train_label_list,
    val_label_list=val_label_list,
    mode='train')
train_loader = paddle.io.DataLoader(
    train_dataset,
    places=PLACE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0)

# val_loader
val_dataset = MyDataset(
    train_img_list=train_image_path_list,
    val_img_list=val_image_path_list,
    train_label_list=train_label_list,
    val_label_list=val_label_list,
    mode='test')
val_loader = paddle.io.DataLoader(
    val_dataset,
    places=PLACE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0)

# 定义模型
class MyNet(paddle.nn.Sequential):
    def __init__(self):
        super(MyNet, self).__init__(
            paddle.vision.models.mobilenet_v2(pretrained=True, scale=1.0),
            paddle.nn.Linear(1000, 4)
        )

# 创建模型实例
model = MyNet()

# # 加载保存的最佳模型参数
# best_model_path = r'E:\PRODUCE\aire\st\work\model\MobileNetV2\50.pdparams'  # 替换为你的最佳模型路径
# model_state_dict = paddle.load(best_model_path)
# model.set_state_dict(model_state_dict)
# E:\PRODUCE\aire\venv\Scripts\python.exe E:\PRODUCE\aire\st\ta.py
# Figure(640x480)
# 训练集大小： 1922
# length of label is  4
# Processing label  : 0
# Num of the label is :  435
# Processing label  : 1
# Num of the label is :  423
# Processing label  : 2
# Num of the label is :  487
# Processing label  : 3
# Num of the label is :  192
# shuffle后数据集大小： 1948
# W0102 18:39:30.107630 16884 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 12.3, Runtime API Version: 11.8
# W0102 18:39:30.145632 16884 gpu_resources.cc:149] device: 0, cuDNN Version: 8.0.
# W0102 18:39:30.609632 16884 gpu_resources.cc:275] WARNING: device:  . The installed Paddle is compiled with CUDNN 8.6, but CUDNN version in your machine is 8.0, which may cause serious incompatible bug. Please recompile or reinstall Paddle with compatible CUDNN version.
# -------------------------------------------------------------------------------
#    Layer (type)         Input Shape          Output Shape         Param #
# ===============================================================================
#      Conv2D-1        [[1, 3, 224, 224]]   [1, 32, 112, 112]         864
#    BatchNorm2D-1    [[1, 32, 112, 112]]   [1, 32, 112, 112]         128
#       ReLU6-1       [[1, 32, 112, 112]]   [1, 32, 112, 112]          0
#      Conv2D-2       [[1, 32, 112, 112]]   [1, 32, 112, 112]         288
#    BatchNorm2D-2    [[1, 32, 112, 112]]   [1, 32, 112, 112]         128
#       ReLU6-2       [[1, 32, 112, 112]]   [1, 32, 112, 112]          0
#      Conv2D-3       [[1, 32, 112, 112]]   [1, 16, 112, 112]         512
#    BatchNorm2D-3    [[1, 16, 112, 112]]   [1, 16, 112, 112]         64
# InvertedResidual-1  [[1, 32, 112, 112]]   [1, 16, 112, 112]          0
#      Conv2D-4       [[1, 16, 112, 112]]   [1, 96, 112, 112]        1,536
#    BatchNorm2D-4    [[1, 96, 112, 112]]   [1, 96, 112, 112]         384
#       ReLU6-3       [[1, 96, 112, 112]]   [1, 96, 112, 112]          0
#      Conv2D-5       [[1, 96, 112, 112]]    [1, 96, 56, 56]          864
#    BatchNorm2D-5     [[1, 96, 56, 56]]     [1, 96, 56, 56]          384
#       ReLU6-4        [[1, 96, 56, 56]]     [1, 96, 56, 56]           0
#      Conv2D-6        [[1, 96, 56, 56]]     [1, 24, 56, 56]         2,304
#    BatchNorm2D-6     [[1, 24, 56, 56]]     [1, 24, 56, 56]          96
# InvertedResidual-2  [[1, 16, 112, 112]]    [1, 24, 56, 56]           0
#      Conv2D-7        [[1, 24, 56, 56]]     [1, 144, 56, 56]        3,456
#    BatchNorm2D-7     [[1, 144, 56, 56]]    [1, 144, 56, 56]         576
#       ReLU6-5        [[1, 144, 56, 56]]    [1, 144, 56, 56]          0
#      Conv2D-8        [[1, 144, 56, 56]]    [1, 144, 56, 56]        1,296
#    BatchNorm2D-8     [[1, 144, 56, 56]]    [1, 144, 56, 56]         576
#       ReLU6-6        [[1, 144, 56, 56]]    [1, 144, 56, 56]          0
#      Conv2D-9        [[1, 144, 56, 56]]    [1, 24, 56, 56]         3,456
#    BatchNorm2D-9     [[1, 24, 56, 56]]     [1, 24, 56, 56]          96
# InvertedResidual-3   [[1, 24, 56, 56]]     [1, 24, 56, 56]           0
#      Conv2D-10       [[1, 24, 56, 56]]     [1, 144, 56, 56]        3,456
#   BatchNorm2D-10     [[1, 144, 56, 56]]    [1, 144, 56, 56]         576
#       ReLU6-7        [[1, 144, 56, 56]]    [1, 144, 56, 56]          0
#      Conv2D-11       [[1, 144, 56, 56]]    [1, 144, 28, 28]        1,296
#   BatchNorm2D-11     [[1, 144, 28, 28]]    [1, 144, 28, 28]         576
#       ReLU6-8        [[1, 144, 28, 28]]    [1, 144, 28, 28]          0
#      Conv2D-12       [[1, 144, 28, 28]]    [1, 32, 28, 28]         4,608
#   BatchNorm2D-12     [[1, 32, 28, 28]]     [1, 32, 28, 28]          128
# InvertedResidual-4   [[1, 24, 56, 56]]     [1, 32, 28, 28]           0
#      Conv2D-13       [[1, 32, 28, 28]]     [1, 192, 28, 28]        6,144
#   BatchNorm2D-13     [[1, 192, 28, 28]]    [1, 192, 28, 28]         768
#       ReLU6-9        [[1, 192, 28, 28]]    [1, 192, 28, 28]          0
#      Conv2D-14       [[1, 192, 28, 28]]    [1, 192, 28, 28]        1,728
#   BatchNorm2D-14     [[1, 192, 28, 28]]    [1, 192, 28, 28]         768
#      ReLU6-10        [[1, 192, 28, 28]]    [1, 192, 28, 28]          0
#      Conv2D-15       [[1, 192, 28, 28]]    [1, 32, 28, 28]         6,144
#   BatchNorm2D-15     [[1, 32, 28, 28]]     [1, 32, 28, 28]          128
# InvertedResidual-5   [[1, 32, 28, 28]]     [1, 32, 28, 28]           0
#      Conv2D-16       [[1, 32, 28, 28]]     [1, 192, 28, 28]        6,144
#   BatchNorm2D-16     [[1, 192, 28, 28]]    [1, 192, 28, 28]         768
#      ReLU6-11        [[1, 192, 28, 28]]    [1, 192, 28, 28]          0
#      Conv2D-17       [[1, 192, 28, 28]]    [1, 192, 28, 28]        1,728
#   BatchNorm2D-17     [[1, 192, 28, 28]]    [1, 192, 28, 28]         768
#      ReLU6-12        [[1, 192, 28, 28]]    [1, 192, 28, 28]          0
#      Conv2D-18       [[1, 192, 28, 28]]    [1, 32, 28, 28]         6,144
#   BatchNorm2D-18     [[1, 32, 28, 28]]     [1, 32, 28, 28]          128
# InvertedResidual-6   [[1, 32, 28, 28]]     [1, 32, 28, 28]           0
#      Conv2D-19       [[1, 32, 28, 28]]     [1, 192, 28, 28]        6,144
#   BatchNorm2D-19     [[1, 192, 28, 28]]    [1, 192, 28, 28]         768
#      ReLU6-13        [[1, 192, 28, 28]]    [1, 192, 28, 28]          0
#      Conv2D-20       [[1, 192, 28, 28]]    [1, 192, 14, 14]        1,728
#   BatchNorm2D-20     [[1, 192, 14, 14]]    [1, 192, 14, 14]         768
#      ReLU6-14        [[1, 192, 14, 14]]    [1, 192, 14, 14]          0
#      Conv2D-21       [[1, 192, 14, 14]]    [1, 64, 14, 14]        12,288
#   BatchNorm2D-21     [[1, 64, 14, 14]]     [1, 64, 14, 14]          256
# InvertedResidual-7   [[1, 32, 28, 28]]     [1, 64, 14, 14]           0
#      Conv2D-22       [[1, 64, 14, 14]]     [1, 384, 14, 14]       24,576
#   BatchNorm2D-22     [[1, 384, 14, 14]]    [1, 384, 14, 14]        1,536
#      ReLU6-15        [[1, 384, 14, 14]]    [1, 384, 14, 14]          0
#      Conv2D-23       [[1, 384, 14, 14]]    [1, 384, 14, 14]        3,456
#   BatchNorm2D-23     [[1, 384, 14, 14]]    [1, 384, 14, 14]        1,536
#      ReLU6-16        [[1, 384, 14, 14]]    [1, 384, 14, 14]          0
#      Conv2D-24       [[1, 384, 14, 14]]    [1, 64, 14, 14]        24,576
#   BatchNorm2D-24     [[1, 64, 14, 14]]     [1, 64, 14, 14]          256
# InvertedResidual-8   [[1, 64, 14, 14]]     [1, 64, 14, 14]           0
#      Conv2D-25       [[1, 64, 14, 14]]     [1, 384, 14, 14]       24,576
#   BatchNorm2D-25     [[1, 384, 14, 14]]    [1, 384, 14, 14]        1,536
#      ReLU6-17        [[1, 384, 14, 14]]    [1, 384, 14, 14]          0
#      Conv2D-26       [[1, 384, 14, 14]]    [1, 384, 14, 14]        3,456
#   BatchNorm2D-26     [[1, 384, 14, 14]]    [1, 384, 14, 14]        1,536
#      ReLU6-18        [[1, 384, 14, 14]]    [1, 384, 14, 14]          0
#      Conv2D-27       [[1, 384, 14, 14]]    [1, 64, 14, 14]        24,576
#   BatchNorm2D-27     [[1, 64, 14, 14]]     [1, 64, 14, 14]          256
# InvertedResidual-9   [[1, 64, 14, 14]]     [1, 64, 14, 14]           0
#      Conv2D-28       [[1, 64, 14, 14]]     [1, 384, 14, 14]       24,576
#   BatchNorm2D-28     [[1, 384, 14, 14]]    [1, 384, 14, 14]        1,536
#      ReLU6-19        [[1, 384, 14, 14]]    [1, 384, 14, 14]          0
#      Conv2D-29       [[1, 384, 14, 14]]    [1, 384, 14, 14]        3,456
#   BatchNorm2D-29     [[1, 384, 14, 14]]    [1, 384, 14, 14]        1,536
#      ReLU6-20        [[1, 384, 14, 14]]    [1, 384, 14, 14]          0
#      Conv2D-30       [[1, 384, 14, 14]]    [1, 64, 14, 14]        24,576
#   BatchNorm2D-30     [[1, 64, 14, 14]]     [1, 64, 14, 14]          256
# InvertedResidual-10  [[1, 64, 14, 14]]     [1, 64, 14, 14]           0
#      Conv2D-31       [[1, 64, 14, 14]]     [1, 384, 14, 14]       24,576
#   BatchNorm2D-31     [[1, 384, 14, 14]]    [1, 384, 14, 14]        1,536
#      ReLU6-21        [[1, 384, 14, 14]]    [1, 384, 14, 14]          0
#      Conv2D-32       [[1, 384, 14, 14]]    [1, 384, 14, 14]        3,456
#   BatchNorm2D-32     [[1, 384, 14, 14]]    [1, 384, 14, 14]        1,536
#      ReLU6-22        [[1, 384, 14, 14]]    [1, 384, 14, 14]          0
#      Conv2D-33       [[1, 384, 14, 14]]    [1, 96, 14, 14]        36,864
#   BatchNorm2D-33     [[1, 96, 14, 14]]     [1, 96, 14, 14]          384
# InvertedResidual-11  [[1, 64, 14, 14]]     [1, 96, 14, 14]           0
#      Conv2D-34       [[1, 96, 14, 14]]     [1, 576, 14, 14]       55,296
#   BatchNorm2D-34     [[1, 576, 14, 14]]    [1, 576, 14, 14]        2,304
#      ReLU6-23        [[1, 576, 14, 14]]    [1, 576, 14, 14]          0
#      Conv2D-35       [[1, 576, 14, 14]]    [1, 576, 14, 14]        5,184
#   BatchNorm2D-35     [[1, 576, 14, 14]]    [1, 576, 14, 14]        2,304
#      ReLU6-24        [[1, 576, 14, 14]]    [1, 576, 14, 14]          0
#      Conv2D-36       [[1, 576, 14, 14]]    [1, 96, 14, 14]        55,296
#   BatchNorm2D-36     [[1, 96, 14, 14]]     [1, 96, 14, 14]          384
# InvertedResidual-12  [[1, 96, 14, 14]]     [1, 96, 14, 14]           0
#      Conv2D-37       [[1, 96, 14, 14]]     [1, 576, 14, 14]       55,296
#   BatchNorm2D-37     [[1, 576, 14, 14]]    [1, 576, 14, 14]        2,304
#      ReLU6-25        [[1, 576, 14, 14]]    [1, 576, 14, 14]          0
#      Conv2D-38       [[1, 576, 14, 14]]    [1, 576, 14, 14]        5,184
#   BatchNorm2D-38     [[1, 576, 14, 14]]    [1, 576, 14, 14]        2,304
#      ReLU6-26        [[1, 576, 14, 14]]    [1, 576, 14, 14]          0
#      Conv2D-39       [[1, 576, 14, 14]]    [1, 96, 14, 14]        55,296
#   BatchNorm2D-39     [[1, 96, 14, 14]]     [1, 96, 14, 14]          384
# InvertedResidual-13  [[1, 96, 14, 14]]     [1, 96, 14, 14]           0
#      Conv2D-40       [[1, 96, 14, 14]]     [1, 576, 14, 14]       55,296
#   BatchNorm2D-40     [[1, 576, 14, 14]]    [1, 576, 14, 14]        2,304
#      ReLU6-27        [[1, 576, 14, 14]]    [1, 576, 14, 14]          0
#      Conv2D-41       [[1, 576, 14, 14]]     [1, 576, 7, 7]         5,184
#   BatchNorm2D-41      [[1, 576, 7, 7]]      [1, 576, 7, 7]         2,304
#      ReLU6-28         [[1, 576, 7, 7]]      [1, 576, 7, 7]           0
#      Conv2D-42        [[1, 576, 7, 7]]      [1, 160, 7, 7]        92,160
#   BatchNorm2D-42      [[1, 160, 7, 7]]      [1, 160, 7, 7]          640
# InvertedResidual-14  [[1, 96, 14, 14]]      [1, 160, 7, 7]           0
#      Conv2D-43        [[1, 160, 7, 7]]      [1, 960, 7, 7]        153,600
#   BatchNorm2D-43      [[1, 960, 7, 7]]      [1, 960, 7, 7]         3,840
#      ReLU6-29         [[1, 960, 7, 7]]      [1, 960, 7, 7]           0
#      Conv2D-44        [[1, 960, 7, 7]]      [1, 960, 7, 7]         8,640
#   BatchNorm2D-44      [[1, 960, 7, 7]]      [1, 960, 7, 7]         3,840
#      ReLU6-30         [[1, 960, 7, 7]]      [1, 960, 7, 7]           0
#      Conv2D-45        [[1, 960, 7, 7]]      [1, 160, 7, 7]        153,600
#   BatchNorm2D-45      [[1, 160, 7, 7]]      [1, 160, 7, 7]          640
# InvertedResidual-15   [[1, 160, 7, 7]]      [1, 160, 7, 7]           0
#      Conv2D-46        [[1, 160, 7, 7]]      [1, 960, 7, 7]        153,600
#   BatchNorm2D-46      [[1, 960, 7, 7]]      [1, 960, 7, 7]         3,840
#      ReLU6-31         [[1, 960, 7, 7]]      [1, 960, 7, 7]           0
#      Conv2D-47        [[1, 960, 7, 7]]      [1, 960, 7, 7]         8,640
#   BatchNorm2D-47      [[1, 960, 7, 7]]      [1, 960, 7, 7]         3,840
#      ReLU6-32         [[1, 960, 7, 7]]      [1, 960, 7, 7]           0
#      Conv2D-48        [[1, 960, 7, 7]]      [1, 160, 7, 7]        153,600
#   BatchNorm2D-48      [[1, 160, 7, 7]]      [1, 160, 7, 7]          640
# InvertedResidual-16   [[1, 160, 7, 7]]      [1, 160, 7, 7]           0
#      Conv2D-49        [[1, 160, 7, 7]]      [1, 960, 7, 7]        153,600
#   BatchNorm2D-49      [[1, 960, 7, 7]]      [1, 960, 7, 7]         3,840
#      ReLU6-33         [[1, 960, 7, 7]]      [1, 960, 7, 7]           0
#      Conv2D-50        [[1, 960, 7, 7]]      [1, 960, 7, 7]         8,640
#   BatchNorm2D-50      [[1, 960, 7, 7]]      [1, 960, 7, 7]         3,840
#      ReLU6-34         [[1, 960, 7, 7]]      [1, 960, 7, 7]           0
#      Conv2D-51        [[1, 960, 7, 7]]      [1, 320, 7, 7]        307,200
#   BatchNorm2D-51      [[1, 320, 7, 7]]      [1, 320, 7, 7]         1,280
# InvertedResidual-17   [[1, 160, 7, 7]]      [1, 320, 7, 7]           0
#      Conv2D-52        [[1, 320, 7, 7]]     [1, 1280, 7, 7]        409,600
#   BatchNorm2D-52     [[1, 1280, 7, 7]]     [1, 1280, 7, 7]         5,120
#      ReLU6-35        [[1, 1280, 7, 7]]     [1, 1280, 7, 7]           0
# AdaptiveAvgPool2D-1  [[1, 1280, 7, 7]]     [1, 1280, 1, 1]           0
#      Dropout-1          [[1, 1280]]           [1, 1280]              0
#      Linear-1           [[1, 1280]]           [1, 1000]          1,281,000
#    MobileNetV2-1     [[1, 3, 224, 224]]       [1, 1000]              0
#      Linear-2           [[1, 1000]]             [1, 4]             4,004
# ===============================================================================
# Total params: 3,542,988
# Trainable params: 3,508,876
# Non-trainable params: 34,112
# -------------------------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 152.88
# Params size (MB): 13.52
# Estimated Total Size (MB): 166.97
# -------------------------------------------------------------------------------
#
# The loss value printed in the log is the current step, and the metric is the average value of previous steps.
# Epoch 1/5
# step 16/16 [==============================] - loss: 0.3359 - acc_top1: 0.9810 - acc_top5: 1.0000 - 10s/step
# save checkpoint at E:\PRODUCE\aire\st\checkpoint\MobileNetV2\0
# Eval begin...
#
# Process finished with exit code -1073741819 (0xC0000005)
#
# The loss value printed in the log is the current step, and the metric is the average value of previous steps.
# Epoch 1/5
# step 10/16 [=================>............] - loss: 1.6458 - acc_top1: 0.5656 - acc_top5: 1.0000 - ETA: 1:27 - 15s/step
# Process finished with exit code -1
model = paddle.Model(model)
model.summary((1, 3, 224, 224))

# 定义优化器
def make_optimizer(parameters=None, momentum=0.9, weight_decay=5e-4, boundaries=None, values=None):
    lr_scheduler = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=boundaries,
        values=values,
        verbose=False)
    lr_scheduler = paddle.optimizer.lr.LinearWarmup(
        learning_rate=0.001,
        warmup_steps=20,
        start_lr=0.001 / 5.,
        end_lr=0.001,
        verbose=False)
    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        weight_decay=weight_decay,
        parameters=parameters)
    return optimizer

base_lr = 0.001
boundaries = [33, 44]
optimizer = make_optimizer(boundaries=boundaries, values=[base_lr, base_lr * 0.1, base_lr * 0.01],
                           parameters=model.parameters())

model.prepare(
    optimizer=optimizer,
    loss=paddle.nn.CrossEntropyLoss(soft_label=True),
    metrics=paddle.metric.Accuracy(topk=(1, 5))
)

# 定义回调函数
visualdl = paddle.callbacks.VisualDL('./visualdl/MobileNetV2')
earlystop = paddle.callbacks.EarlyStopping(
    'acc',
    mode='max',
    patience=2,
    verbose=1,
    min_delta=0,
    baseline=None,
    save_best_model=True)

# 继续训练模型
model.fit(
    train_loader,
    val_loader,
    epochs=5,
    save_freq=1,
    save_dir='checkpoint/MobileNetV2',
    callbacks=[visualdl, earlystop],
    verbose=1
)

# 保存训练后的模型
model.save('work/model/best_model')  # 保存训练好的模型