import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
import cv2

# 定义模型
class MyNet(nn.Sequential):
    def __init__(self):
        super(MyNet, self).__init__(
            paddle.vision.models.mobilenet_v2(pretrained=True, scale=1.0),
            nn.Linear(1000, 4)
        )

# 加载模型
model = MyNet()
model_path = r'D:\ultralytics-main\zzzaaa_project\hd\model\best_model\75.pdparams'  # 替换为你的模型路径
model.load_dict(paddle.load(model_path))
model.eval()

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
    if isinstance(image_path, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))  # 将BGR转换为RGB
    else:
        image = Image.open(image_path).convert('RGB')
    image = data_transforms(image)
    image = paddle.to_tensor(image).unsqueeze(0)  # 添加批次维度
    return image

# 测试单张图片
def te_single_image(image_path):
    image = load_and_preprocess_image(image_path)
    with paddle.no_grad():
        output = model(image)
    predicted_class = paddle.argmax(output, axis=1).item()
    predicted_probabilities = paddle.nn.functional.softmax(output, axis=1).numpy()[0]
    return predicted_class, predicted_probabilities

# 检测函数
def detect(source):
    if source.startswith('http') or source.endswith(('.mp4', '.avi')):
        # 视频检测
        cap = cv2.VideoCapture(source)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = load_and_preprocess_image(frame)
            with paddle.no_grad():
                output = model(frame)
            predicted_class = paddle.argmax(output, axis=1).item()
            predicted_probabilities = paddle.nn.functional.softmax(output, axis=1).numpy()[0]
            # 绘制预测结果
            for i, prob in enumerate(predicted_probabilities):
                cv2.putText(frame, f'Class {i}: {prob:.4f}', (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    elif source.startswith('camera'):
        # 摄像头检测
        cap = cv2.VideoCapture(0)  # 0 是默认摄像头
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = load_and_preprocess_image(frame)
            with paddle.no_grad():
                output = model(frame)
            predicted_class = paddle.argmax(output, axis=1).item()
            predicted_probabilities = paddle.nn.functional.softmax(output, axis=1).numpy()[0]
            # 绘制预测结果
            for i, prob in enumerate(predicted_probabilities):
                cv2.putText(frame, f'Class {i}: {prob:.4f}', (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        # 图片检测
        predicted_class, predicted_probabilities = te_single_image(source)
        # 打印预测结果
        print(f'Predicted class: {predicted_class}')
        print(f'Predicted probabilities: {predicted_probabilities}')
        # 打印每个类别的预测概率
        for i, prob in enumerate(predicted_probabilities):
            print(f'Class {i}: Probability {prob:.4f}')

# 使用示例
# detect('path/to/your/image.jpg')  # 图片检测
# detect('path/to/your/video.mp4')  # 视频检测
detect('camera')  # 摄像头检测