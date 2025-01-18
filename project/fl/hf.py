import os
import random
import shutil
from sklearn.model_selection import train_test_split
import yaml

# 数据集的根目录
dataset_root = './dataset'  # 请根据你的实际路径修改

# 图像和标签文件夹路径
images_dir = os.path.join(dataset_root, 'images')
labels_dir = os.path.join(dataset_root, 'labels')

# 类别名称文件路径
classes_file = os.path.join(labels_dir, 'classes.txt')

# 读取类别名称
with open(classes_file, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# 划分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 确保比例之和为1
assert train_ratio + val_ratio + test_ratio == 1, "The sum of train_ratio, val_ratio, and test_ratio must be 1."

# 获取所有图像文件的列表
image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f)) and f.endswith(('.jpg', '.png'))]

# 随机打乱图像文件列表
random.shuffle(image_files)

# 划分训练集和非训练集（验证集+测试集）
train_files, temp_files = train_test_split(image_files, test_size=1 - train_ratio, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

# 创建训练、验证和测试文件夹
train_images_dir = os.path.join(dataset_root, 'images', 'train')
val_images_dir = os.path.join(dataset_root, 'images', 'val')
test_images_dir = os.path.join(dataset_root, 'images', 'test')

train_labels_dir = os.path.join(dataset_root, 'labels', 'train')
val_labels_dir = os.path.join(dataset_root, 'labels', 'val')
test_labels_dir = os.path.join(dataset_root, 'labels', 'test')

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# 复制图像文件和对应的标签文件到对应的文件夹
def copy_files_and_labels(image_list, dest_images_dir, dest_labels_dir, src_images_dir, src_labels_dir):
    for file in image_list:
        image_path = os.path.join(src_images_dir, file)
        label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')  # 假设标签文件名与图像文件名相对应
        shutil.copy(image_path, os.path.join(dest_images_dir, file))
        shutil.copy(os.path.join(src_labels_dir, label_file), os.path.join(dest_labels_dir, label_file))

copy_files_and_labels(train_files, train_images_dir, train_labels_dir, images_dir, labels_dir)
copy_files_and_labels(val_files, val_images_dir, val_labels_dir, images_dir, labels_dir)
copy_files_and_labels(test_files, test_images_dir, test_labels_dir, images_dir, labels_dir)

# 创建data.yaml文件
data_config = {
    'path': dataset_root,
    'train': 'labels/train',
    'val': 'labels/val',
    'test': 'labels/test',
    'nc': len(class_names),
    'names': class_names
}

with open(os.path.join(dataset_root, 'labels', 'data.yaml'), 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print('数据集划分完成，data.yaml 文件已生成。')