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

# 加载保存的最佳模型参数
best_model_path = r'/home/aistudio/checkpoint/MobileNetV2/final.pdparams'  # 替换为你的最佳模型路径
model_state_dict = paddle.load(best_model_path)
model.set_state_dict(model_state_dict)

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
    epochs=50,
    save_freq=1,
    save_dir='checkpoint2/MobileNetV2',
    callbacks=[visualdl, earlystop],
    verbose=1
)

# 保存训练后的模型
model.save('work/model2/best_model2')  # 保存训练好的模型