import paddle

import paddle.nn as nn
import paddle.vision.transforms as T
from PIL import Image
import numpy as np

# 定义模型
class MyNet(paddle.nn.Sequential):
    def __init__(self):
        super(MyNet, self).__init__(
            paddle.vision.models.mobilenet_v2(pretrained=True, scale=1.0),
            paddle.nn.Linear(1000, 4)
        )

# 加载模型
model = MyNet()
model_path = r'E:\PRODUCE\aire\st\work\model\best_model\15.pdparams'
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
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image)
    image = paddle.to_tensor(image).unsqueeze(0)  # 添加批次维度
    return image

# 测试单张图片
def test_single_image(image_path):
    image = load_and_preprocess_image(image_path)
    with paddle.no_grad():
        output = model(image)
    predicted_class = paddle.argmax(output, axis=1).item()
    predicted_probabilities = paddle.nn.functional.softmax(output, axis=1).numpy()[0]
    return predicted_class, predicted_probabilities

# 测试图片路径
image_path = r'E:\PRODUCE\aire\st\data\test_data\AORmR.jpg'  # 替换为你要测试的图片路径
predicted_class, predicted_probabilities = test_single_image(image_path)

# 打印预测结果
print(f'Predicted class: {predicted_class}')
print(f'Predicted probabilities: {predicted_probabilities}')

# 打印每个类别的预测概率
for i, prob in enumerate(predicted_probabilities):
    print(f'Class {i}: Probability {prob:.4f}')


