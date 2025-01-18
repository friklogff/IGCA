import gradio as gr
from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 定义模型路径
model_paths = {
    "Model 1 (best520.pt)": r'D:\ultralytics-main\zzzaaa_project\fl\model\best-zh.pt',
    "Model 2 (best900.pt)": r'D:\ultralytics-main\zzzaaa_project\shinei-zh\model\best900.pt',
    "Model 3 (voc_best_220.pt)": r'D:\ultralytics-main\zzzaaa_project\voc\voc_best_220.pt'
}

# 加载模型
models = {name: YOLO(model=path) for name, path in model_paths.items()}

# 默认保存路径
default_save_dir = Path(r'D:\ultralytics-main\zzzaaa_project\gui\test')
def custom_detect(image_path, model_name, conf_threshold=0.5):
    """
    自定义检测函数，进行图像标注并保存结果到指定路径.

    Args:
        image_path (str): 要进行检测的图像路径.
        model_name (str): 模型名称.
        conf_threshold (float, optional): 置信度阈值. 默认为0.5.
    """
    # 获取模型
    model = models[model_name]

    # 进行预测
    results = model(image_path)

    # 读取图像
    img = cv2.imread(image_path)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("simhei.ttf", 20)  # 使用黑体字体，需要确保字体文件存在

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

    # 将 PIL 图像转换回 OpenCV 格式
    img = cv2.cvtColor(np.array(img_pil_with_boxes), cv2.COLOR_RGB2BGR)

    # 保存标注后的图像
    save_path = default_save_dir / Path(image_path).name
    default_save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)

    return str(save_path)
# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# YOLO Object Detection")
    gr.Markdown("选择一个模型并上传图像进行目标检测。检测结果将保存在指定目录中。")

    with gr.Column():
        with gr.Row():
            image_input = gr.Image(type="filepath", label="上传图像")
            model_dropdown = gr.Dropdown(choices=list(model_paths.keys()), value=list(model_paths.keys())[0], label="选择模型")
        with gr.Row():
            conf_threshold_input = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="置信度阈值")
            detect_button = gr.Button("检测")
        result_image = gr.Image(type="filepath", label="检测结果")

    detect_button.click(
        fn=custom_detect,
        inputs=[image_input, model_dropdown, conf_threshold_input],
        outputs=result_image
    )

# 启动应用
demo.launch(share=True)
