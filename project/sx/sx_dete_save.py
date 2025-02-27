import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def draw_text_with_pillow(image, text, position, font_path, font_size, color):
    """
    使用 Pillow 绘制中文文本，并返回 OpenCV 的图像格式。
    参数:
        image: OpenCV 图像
        text: 要绘制的文本
        position: 文本位置 (x, y)
        font_path: 字体文件路径
        font_size: 字体大小
        color: 文本颜色 (B, G, R)
    """
    # 将 OpenCV 图像转换为 PIL 图像
    img_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pillow)

    # 加载字体
    font = ImageFont.truetype(font_path, font_size)

    # 绘制文本
    draw.text(position, text, font=font, fill=color[::-1])  # 注意颜色顺序为 RGB

    # 转换回 OpenCV 图像
    return cv2.cvtColor(np.array(img_pillow), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    # 加载模型
    model = YOLO(model=r'D:\ultralytics-main\zzzaaa_project\sx\weights\best_sx.pt')

    # 打开视频
    cap = cv2.VideoCapture('test.mp4')

    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率

    # 创建 VideoWriter 对象，用于保存视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置视频编码格式
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))  # 输出文件名、编码格式、帧率、分辨率

    # 字体路径（需要指定一个支持中文的字体文件）
    font_path = r'C:\Windows\Fonts\simhei.ttf'  # 示例路径，可根据实际情况修改

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 进行预测
        results = model.predict(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                class_id = int(box.cls.item())
                conf = box.conf.item()
                label = f'{model.names[class_id]}: {conf:.2f}'

                # 使用 Pillow 绘制中文文本
                frame = draw_text_with_pillow(frame, label, (x1, y1 - 20), font_path, font_size=16, color=(0, 255, 0))

        # 写入帧到输出视频
        out.write(frame)

        cv2.imshow('YOLO Real-time Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # 释放 VideoWriter 对象
    cv2.destroyAllWindows()