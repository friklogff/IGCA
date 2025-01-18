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
train_images = pd.read_csv('data/train_data/train_label.csv', usecols=['image_name','label'])

# label shuffling，定义标签打乱模块
def labelShuffling(dataFrame, groupByName='label'):
    groupDataFrame = dataFrame.groupby(by=[groupByName])
    labels = groupDataFrame.size()
    print("length of label is ", len(labels))
    maxNum = max(labels)
    lst = pd.DataFrame()
    for i in range(len(labels)):
        print("Processing label  :", i)
        tmpGroupBy = groupDataFrame.get_group(i)
        createdShuffleLabels = np.random.permutation(np.array(range(maxNum))) % labels[i]
        print("Num of the label is : ", labels[i])
        lst = pd.concat([lst, tmpGroupBy.iloc[createdShuffleLabels]], ignore_index=True)
    return lst

# 进行训练集和测试集划分，按照先打乱后划分原则
all_size = len(train_images)
print("训练集大小：", all_size)
train_size = int(all_size * 0.8)
train_image_list = train_images[:train_size]
val_image_list = train_images[train_size:]



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
    T.Transpose(),
    T.Normalize(
        mean=[127.5, 127.5, 127.5],
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
        return data_transforms(image), paddle.nn.functional.label_smooth(label)

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

# 批量评估模型
model_dir = 'E:/PRODUCE/aire/st/work/model/MobileNetV2'
model_paths = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pdparams')]

results = []

for model_path in model_paths:
    model = MyNet()
    model = paddle.Model(model)
    model.summary((1, 3, 224, 224))

    # 加载模型
    model.load(model_path)

    # 准备模型
    model.prepare(
        loss=paddle.nn.CrossEntropyLoss(soft_label=True),
        metrics=paddle.metric.Accuracy()
    )

    # 评估模型
    result = model.evaluate(val_loader, batch_size=1, verbose=1)
    results.append({'model_path': model_path, 'accuracy': result['acc'], 'loss': result['loss']})

# 格式化记录结果
results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)

print(results_df)
