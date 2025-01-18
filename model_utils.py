import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from ultralytics import YOLO

# 定义分类模型
class MyNet(paddle.nn.Sequential):
    def __init__(self):
        super(MyNet, self).__init__(
            paddle.vision.models.mobilenet_v2(pretrained=True, scale=1.0),
            paddle.nn.Linear(1000, 4)
        )

# 加载分类模型
def load_classification_model(model_path):
    model = MyNet()
    model.load_dict(paddle.load(model_path))
    model.eval()
    return model

# 定义数据预处理
data_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.Transpose(),    # HWC -> CHW
    T.Normalize(
        mean=[127.5, 127.5, 127.5],        # 归一化
        std=[127.5, 127.5, 127.5],
        to_rgb=True)
])

# 加载并预处理图片
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image)
    image = paddle.to_tensor(image).unsqueeze(0)  # 添加批次维度
    return image

# 测试单张图片
def classify_image(model, image_path):
    image = load_and_preprocess_image(image_path)
    with paddle.no_grad():
        output = model(image)
    predicted_class = paddle.argmax(output, axis=1).item()
    predicted_probabilities = paddle.nn.functional.softmax(output, axis=1).numpy()[0]
    return predicted_class, predicted_probabilities

# 定义污染级别标签
pollution_levels = {
    0: "良好：0级污染河道",
    1: "轻度污染：1级污染河道",
    2: "中度污染：2级污染河道",
    3: "严重污染：3级污染河道"
}

# 加载 YOLO 模型
def load_yolo_models(model_paths):
    return {name: YOLO(model=path) for name, path in model_paths.items()}

# 进行目标检测并标注
def detect_and_annotate(image_path, model, conf_threshold=0.5, font_path="simhei.ttf"):
    results = model(image_path)
    img = cv2.imread(image_path)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, 20)  # 使用黑体字体，需要确保字体文件存在

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < conf_threshold:
                continue  # 如果置信度低于阈值，则跳过

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # 转换为整数并转换为列表

            # 绘制矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 将 OpenCV 图像上的矩形框合并到 PIL 图像中
    img_with_boxes = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil_with_boxes = Image.fromarray(img_with_boxes)
    draw = ImageDraw.Draw(img_pil_with_boxes)

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < conf_threshold:
                continue  # 如果置信度低于阈值，则跳过

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # 转换为整数并转换为列表

            # 获取类别名称
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # 绘制文本
            text = f"{class_name} {confidence:.2f}"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 确保文本不超出图片边界
            text_x = max(x1, 0)
            text_y = max(y1 - text_height - 4, 0)

            # 绘制文本背景框
            draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height + 4], fill=(0, 255, 0))
            draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

    return img_pil_with_boxes
