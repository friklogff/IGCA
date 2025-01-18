import os
import shutil
import random

def split_dataset(dataset_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # 创建目标文件夹结构
    images_path = os.path.join(output_path, 'images')
    labels_path = os.path.join(output_path, 'labels')
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # 创建训练集、验证集和测试集的子文件夹
    train_images_path = os.path.join(images_path, 'train')
    val_images_path = os.path.join(images_path, 'val')
    test_images_path = os.path.join(images_path, 'test')
    train_labels_path = os.path.join(labels_path, 'train')
    val_labels_path = os.path.join(labels_path, 'val')
    test_labels_path = os.path.join(labels_path, 'test')

    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(test_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)
    os.makedirs(test_labels_path, exist_ok=True)

    # 获取所有图片和对应的txt文件
    all_files = os.listdir(dataset_path)
    image_files = [f for f in all_files if f.endswith('.jpg') or f.endswith('.png')]
    label_files = [f for f in all_files if f.endswith('.txt')]

    # 确保图片和标签文件匹配
    matched_files = []
    for image_file in image_files:
        label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        if label_file in label_files:
            matched_files.append((image_file, label_file))

    assert len(matched_files) > 0, "没有找到匹配的图片和标签文件"

    # 随机打乱文件列表
    random.shuffle(matched_files)

    # 计算划分的数量
    total_files = len(matched_files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)

    # 划分数据集
    train_files = matched_files[:train_size]
    val_files = matched_files[train_size:train_size + val_size]
    test_files = matched_files[train_size + val_size:]

    # 移动文件到对应的子文件夹
    def move_files(file_list, src_path, dest_images_path, dest_labels_path):
        for image_file, label_file in file_list:
            image_path = os.path.join(src_path, image_file)
            label_path = os.path.join(src_path, label_file)
            shutil.move(image_path, dest_images_path)
            shutil.move(label_path, dest_labels_path)

    move_files(train_files, dataset_path, train_images_path, train_labels_path)
    move_files(val_files, dataset_path, val_images_path, val_labels_path)
    move_files(test_files, dataset_path, test_images_path, test_labels_path)

    print("数据集划分完成！")

# 指定原始数据集路径和输出路径
dataset_path = 'yolo'  # 替换为你的数据集路径
output_path = 'yolo/output'  # 替换为你希望输出的路径

# 调用函数进行数据集划分
split_dataset(dataset_path, output_path)