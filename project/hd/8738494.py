#!/usr/bin/env python
# coding: utf-8

# # 水体污染等级分类

# ### 1、解压数据集
# 1. 数据集格式：train中有图片和对应标注好标签和污染等级的csv文件；；test文件只有图片。
# 1. train的数据集8:2划分为训练集和验证集
# 

# In[ ]:


get_ipython().system('pwd   #显示目录')
get_ipython().system('unzip -oq data/data101229/data.zip')


# ### 2、导入库

# In[ ]:


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


# ### 3、EDA（Exploratory Data Analysis）与数据预处理
# 数据EDA，这里是提供思路，非必须
# 
# &emsp;&emsp;探索性数据分析（Exploratory Data Analysis，简称EDA），是指对已有的数据（原始数据）进行分析探索，通过作图、制表、方程拟合、计算特征量等手段探索数据的结构和规律的一种数据分析方法。一般来说，我们最初接触到数据的时候往往是毫无头绪的，不知道如何下手，这时候探索性数据分析就非常有效。
# 
# &emsp;&emsp;对于图像分类任务，我们通常首先应该统计出每个类别的数量，查看训练集的数据分布情况。通过数据分布情况分析赛题，形成解题思路。（洞察数据的本质很重要。）

# In[ ]:


# 数据EDA
df = pd.read_csv('data/train_data/train_label.csv')
d = df['label'].hist().get_figure()
# d.savefig('2.jpg')
print(d)


# In[ ]:


# 读取数据
train_images = pd.read_csv('data/train_data/train_label.csv', usecols=['image_name','label'])  # 读取文件名和类别
# print(train_images)

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
        lst=lst.append(tmpGroupBy.iloc[createdShuffleLabels], ignore_index=True)
        # print("Done")
    # lst.to_csv('test1.csv', index=False)
    return lst

#####由于数据集是按类规则排放的，应该先打乱再划分吧？？？？？？？
"""
df = labelShuffling(train_images)
df = shuffle(df)
# 划分训练集和验证集 8:2;;后面好像测试集直接引用了验证集的图
all_size = len(train_images)
print("训练集大小：", all_size)
train_size = int(all_size * 0.8)
train_image_list = df[:train_size]
val_image_list = df[train_size:]
"""
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


# In[ ]:


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


# In[ ]:


# 构建Dataset
#需要用GPU跑
import paddle
class MyDataset(paddle.io.Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, train_img_list, val_img_list, train_label_list, val_label_list, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集;;
        ### 应该是验证集吧；直接将训练集数据分成了训练和验证，还得想想怎么测试测试集的图片
        """
        super(MyDataset, self).__init__()
        self.img = []
        self.label = []
        # 借助pandas读csv的库
        self.train_images = train_img_list
        self.val_images = val_img_list  ###############
        self.train_label = train_label_list
        self.val_label = val_label_list  ##############
        if mode == 'train':
            # 读train_images的数据
            for img,la in zip(self.train_images, self.train_label):
                self.img.append('data/train_data/train_image/' + img)
                self.label.append(la)
        else:
            # 读test_images的数据
            for img,la in zip(self.val_images, self.val_label):   #############
                self.img.append('data/train_data/train_image/' + img)
                self.label.append(la)

         

    def load_img(self, image_path):
        # 实际使用时使用Pillow相关库进行图片读取即可，这里我们对数据先做个模拟
        image = Image.open(image_path).convert('RGB')
        return image

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        image = self.load_img(self.img[index])
        label = self.label[index]
        # label = paddle.to_tensor(label)
        
        return data_transforms(image), paddle.nn.functional.label_smooth(label)

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
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


# In[ ]:


# 定义模型
class MyNet(paddle.nn.Sequential):
    def __init__(self):
        super(MyNet, self).__init__(
                paddle.vision.models.mobilenet_v2(pretrained=True, scale=1.0),
                paddle.nn.Linear(1000, 4)
        )

model = MyNet()
model = paddle.Model(model)
model.summary((1, 3, 224, 224))


# ### 模型训练 Trick
# 具体内容见原项目

# In[10]:


def make_optimizer(parameters=None, momentum=0.9, weight_decay=5e-4, boundaries=None, values=None):
    
    lr_scheduler = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=boundaries, 
        values=values,
        verbose=False)

    lr_scheduler = paddle.optimizer.lr.LinearWarmup(
        learning_rate=base_lr,
        warmup_steps=20,
        start_lr=base_lr / 5.,
        end_lr=base_lr,
        verbose=False)

    # optimizer = paddle.optimizer.Momentum(
    #     learning_rate=lr_scheduler,
    #     weight_decay=weight_decay,
    #     momentum=momentum,
    #     parameters=parameters)

    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        weight_decay=weight_decay,
        parameters=parameters)

    return optimizer


base_lr = 0.001
boundaries = [33, 44]

optimizer = make_optimizer(boundaries=boundaries, values=[base_lr, base_lr*0.1, base_lr*0.01], parameters=model.parameters())

model.prepare(
    optimizer=optimizer,
    loss=paddle.nn.CrossEntropyLoss(soft_label=True),
    metrics=paddle.metric.Accuracy(topk=(1, 5))
)

# callbacks
visualdl = paddle.callbacks.VisualDL('./visualdl/MobileNetV2')
earlystop = paddle.callbacks.EarlyStopping( # acc不在上升时停止
    'acc',
    mode='max',
    patience=2,
    verbose=1,
    min_delta=0,
    baseline=None,
    save_best_model=True)

model.fit(
    train_loader,
    val_loader,
    epochs=100,
    save_freq=5,
    save_dir='checkpoint/MobileNetV2',
    callbacks=[visualdl, earlystop],
    verbose=1
)
# 训练模型保存
model.save('work/model/best_model')  # save for training


# 在你的代码中，自动保存最佳模型是通过`EarlyStopping`回调函数实现的。`EarlyStopping`回调函数会在训练过程中监控指定的指标（在你的代码中是`acc`，即准确率），并在指标不再提升时停止训练。同时，它会自动保存性能最佳的模型。
# 
# ### 自动保存的位置
# 自动保存的模型会存储在`save_dir`指定的目录中。在你的代码中，`save_dir`设置为`'checkpoint/MobileNetV2'`：
# 
# ```python
# model.fit(
#     train_loader,
#     val_loader,
#     epochs=50,
#     save_freq=5,
#     save_dir='checkpoint/MobileNetV2',  # 保存模型的目录
#     callbacks=[visualdl, earlystop],
#     verbose=1
# )
# ```
# 
# ### 详细解释
# 1. **EarlyStopping回调函数**：
#    - `EarlyStopping`回调函数会监控指定的指标（在你的代码中是`acc`，即准确率）。
#    - 当指标不再提升时，`EarlyStopping`回调函数会停止训练。
#    - 同时，它会自动保存性能最佳的模型。
# 
# 2. **save_best_model参数**：
#    - `save_best_model=True`参数确保了在训练过程中保存性能最佳的模型。
# 
# 3. **save_dir参数**：
#    - `save_dir`参数指定了保存模型的目录。在你的代码中，`save_dir`设置为`'checkpoint/MobileNetV2'`。
# 
# ### 示例代码
# 以下是你的代码中相关部分的示例：
# 
# ```python
# # 定义EarlyStopping回调函数
# earlystop = paddle.callbacks.EarlyStopping(
#     'acc',  # 监控的指标
#     mode='max',  # 指标的变化方向，'max'表示指标越大越好
#     patience=2,  # 在指标不再提升的情况下，等待的epoch数
#     verbose=1,  # 是否打印日志
#     min_delta=0,  # 指标变化的最小阈值
#     baseline=None,  # 初始指标值
#     save_best_model=True  # 是否保存最佳模型
# )
# 
# # 训练模型
# model.fit(
#     train_loader,
#     val_loader,
#     epochs=50,
#     save_freq=5,
#     save_dir='checkpoint/MobileNetV2',  # 保存模型的目录
#     callbacks=[visualdl, earlystop],
#     verbose=1
# )
# ```
# 
# ### 总结
# 通过设置`EarlyStopping`回调函数的`save_best_model=True`参数，你可以确保在训练过程中保存性能最佳的模型。保存的模型会存储在`save_dir`指定的目录中，例如`'checkpoint/MobileNetV2'`。这样，你就可以在训练完成后使用最佳模型进行推理或进一步的评估。

# In[ ]:


# 训练模型保存
model.save('work/model/best_model')  # save for training


# ### 模型评估

# In[ ]:


model.load('work/model/best_model')
model.prepare(
    loss=paddle.nn.CrossEntropyLoss(soft_label=True),
    metrics=paddle.metric.Accuracy()
)
result = model.evaluate(val_loader, batch_size=1, verbose=1)
print(result)


# In[ ]:


model.save('work/model/best_model', training=False)  # save for inference

