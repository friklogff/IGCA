import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
import cv2

# 定义模型
class MyNet(paddle.nn.Sequential):
    def __init__(self):
        super(MyNet, self).__init__(
            paddle.vision.models.mobilenet_v2(pretrained=True, scale=1.0),
            paddle.nn.Linear(1000, 4)
        )

# 加载模型
model = MyNet()
model_path = r'D:\ultralytics-main\zzzaaa_project\hd\model\best_model\75.pdparams'
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
def tt_single_image(image_path):
    image = load_and_preprocess_image(image_path)
    with paddle.no_grad():
        output = model(image)
    predicted_class = paddle.argmax(output, axis=1).item()
    predicted_probabilities = paddle.nn.functional.softmax(output, axis=1).numpy()[0]
    return predicted_class, predicted_probabilities

# 测试图片路径
image_path = r'E:\PRODUCE\aire\st\data\test_data\AORmR.jpg'  # 替换为你要测试的图片路径
predicted_class, predicted_probabilities = tt_single_image(image_path)

# 打印预测结果
print(f'Predicted class: {predicted_class}')
print(f'Predicted probabilities: {predicted_probabilities}')

# 打印每个类别的预测概率
for i, prob in enumerate(predicted_probabilities):
    print(f'Class {i}: Probability {prob:.4f}')

# 定义污染级别标签
pollution_levels = {
    0: "良好：0级污染河道",
    1: "轻度污染：1级污染河道",
    2: "中度污染：2级污染河道",
    3: "严重污染：3级污染河道"
}

# 加载原始图片
original_image = cv2.imread(image_path)

# 在图片上标注污染级别
label = pollution_levels.get(predicted_class, "Unknown")

# 使用支持中文的字体文件
font_path = r'C:\Users\admin\Downloads\Fonts\simhei.ttf'  # 替换为你下载的字体文件路径
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0)
thickness = 2

# 使用PIL加载字体
from PIL import ImageFont
font = ImageFont.truetype(font_path, 30)

# 将图片转换为PIL格式
pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

# 使用PIL在图片上绘制中文
draw = ImageDraw.Draw(pil_image)
draw.text((10, 30), label, font=font, fill=(0, 255, 0))

# 将图片转换回OpenCV格式
annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# 显示标注后的图片
cv2.imshow("Pollution Level", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
