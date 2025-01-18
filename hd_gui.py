import gradio as gr
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model_utils import load_classification_model, classify_image, pollution_levels, load_yolo_models, detect_and_annotate

# 定义模型路径
model_paths = {
    "Model 1 (best520.pt)": r'D:\ultralytics-main\zzzaaa_project\fl\model\best-zh.pt',
    "Model 2 (best900.pt)": r'D:\ultralytics-main\zzzaaa_project\shinei-zh\model\best900.pt',
    "Model 3 (voc_best_220.pt)": r'D:\ultralytics-main\zzzaaa_project\voc\voc_best_220.pt',
    "Classification Model": r'D:\ultralytics-main\zzzaaa_project\hd\model\best_model\75.pdparams'
}

# 加载模型
models = load_yolo_models(model_paths)
classification_model = load_classification_model(model_paths["Classification Model"])

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
    if model_name == "Classification Model":
        # 进行图像分类
        predicted_class, predicted_probabilities = classify_image(classification_model, image_path)
        label = pollution_levels.get(predicted_class, "Unknown")

        # 加载原始图片
        img = cv2.imread(image_path)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("simhei.ttf", 30)  # 使用黑体字体，需要确保字体文件存在

        # 使用PIL在图片上绘制中文
        draw.text((10, 30), label, font=font, fill=(0, 255, 0))

        # 将 PIL 图像转换回 OpenCV 格式
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 保存标注后的图像
        save_path = default_save_dir / Path(image_path).name
        default_save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)

        return str(save_path)
    else:
        # 获取模型
        model = models[model_name]

        # 进行目标检测并标注
        img_pil_with_boxes = detect_and_annotate(image_path, model, conf_threshold)

        # 将 PIL 图像转换回 OpenCV 格式
        img = cv2.cvtColor(np.array(img_pil_with_boxes), cv2.COLOR_RGB2BGR)

        # 保存标注后的图像
        save_path = default_save_dir / Path(image_path).name
        default_save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)

        return str(save_path)

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# YOLO Object Detection and Image Classification")
    gr.Markdown("选择一个模型并上传图像进行目标检测或图像分类。结果将保存在指定目录中。")

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
