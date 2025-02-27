import cv2
from ultralytics import YOLO, solutions
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
    heatmap = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, show=True, model="yolo11n.pt")

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        im0 = heatmap.generate_heatmap(im0)
        cv2.imshow("Heatmap", im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()