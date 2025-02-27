import cv2
import os


def images_to_video(image_folder, output_video_path, fps=30):
    """
    将图片文件夹中的图片按名称顺序合成视频。

    参数:
        image_folder (str): 包含图片的文件夹路径。
        output_video_path (str): 输出视频的路径。
        fps (int): 输出视频的帧率，默认为30。
    """
    # 获取图片文件夹中的所有图片文件
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # 按名称排序
    images.sort()

    # 获取第一张图片的尺寸
    if not images:
        print("没有找到图片文件，请检查文件夹路径！")
        return

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 将图片写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放视频写入对象
    video.release()
    print(f"视频已成功保存到 {output_video_path}")


# 示例用法
image_folder = r'D:\ultralytics-main\zzzaaa_project\sx\trash-icra19\yolo_dataset\images\train'  # 替换为你的图片文件夹路径
output_video_path = 'test.mp4'  # 输出视频路径
fps = 9  # 帧率
images_to_video(image_folder, output_video_path, fps)