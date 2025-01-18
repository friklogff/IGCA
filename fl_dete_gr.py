import gradio as gr
from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import torch
# D:\ultralytics-main\zzzaaa_project\fl\model\best520.pt
# D:\ultralytics-main\zzzaaa_project\shinei-zh\model\best900.pt
# D:\ultralytics-main\zzzaaa_project\voc\best_60.pt
# 加载模型
model = YOLO(model=r'D:\ultralytics-main\zzzaaa_project\fl\model\best520.pt')

def custom_detect(image_path=r'D:\ultralytics-main\zzzaaa_project\gui\test', conf_threshold=0.5):
    """
    自定义检测函数，进行图像标注并保存结果到当前工作目录.

    Args:
        image_path (str): 要进行检测的图像路径.
        conf_threshold (float, optional): 置信度阈值. 默认为0.5.
    """
    # 获取当前工作目录
    save_dir = Path.cwd()

    # 确保保存目录存在
    save_dir.mkdir(parents=True, exist_ok=True)

    # 进行预测
    results = model(image_path)

    # 获取检测结果
    for result in results:
        # 绘制边界框和标注
        img = cv2.imread(image_path)
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < conf_threshold:
                continue  # 如果置信度低于阈值，则跳过

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # 转换为整数并转换为列表
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 获取类别名称
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # 绘制文本
            text = f"{class_name} {confidence:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(img, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)
            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 保存标注后的图像
        save_path = save_dir / Path(image_path).name
        cv2.imwrite(str(save_path), img)

    return str(save_path)

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# YOLO Object Detection")
    gr.Markdown("Upload an image to detect objects. The detection result will be saved in the current working directory.")

    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload Image")
        conf_threshold_input = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Confidence Threshold")

    with gr.Row():
        detect_button = gr.Button("Detect")
        result_image = gr.Image(type="filepath", label="Detection Result")

    detect_button.click(fn=custom_detect, inputs=[image_input, conf_threshold_input], outputs=result_image)

# 启动应用
demo.launch()