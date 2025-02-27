import argparse
import functools
import librosa
import numpy as np
import gradio as gr
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
from model_utils import load_classification_model, classify_image, pollution_levels, load_yolo_models, \
    detect_and_annotate
from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments

# ========================== 模型配置 ==========================
model_paths = {
    # 视觉模型
    "目标检测模型1": r'D:\ultralytics-main\zzzaaa_project\fl\model\best-zh.pt',
    "室内检测模型": r'D:\ultralytics-main\zzzaaa_project\shinei-zh\model\best900.pt',
    "VOC通用模型": r'D:\ultralytics-main\zzzaaa_project\voc\voc_best_220.pt',
    "污染分类模型": r'D:\ultralytics-main\zzzaaa_project\hd\model\best_model\75.pdparams',

    # 音频模型
    "音频分类模型": 'models/CAMPPlus_Fbank/best_model/'
}


# ========================== 模型加载 ==========================
def load_all_models():
    """加载所有预训练模型"""
    print("正在加载视觉模型...")
    vision_models = load_yolo_models({
        k: v for k, v in model_paths.items() if "模型" in k and "音频" not in k
    })

    print("加载污染分类模型...")
    pollution_model = load_classification_model(model_paths["污染分类模型"])

    print("初始化音频分类器...")
    audio_parser = argparse.ArgumentParser()
    add_arg = functools.partial(add_arguments, argparser=audio_parser)
    add_arg('configs', str, 'configs/cam++.yml', '配置文件路径')
    add_arg('use_gpu', bool, True, '是否使用GPU')
    audio_args = audio_parser.parse_args(args=[])
    audio_predictor = MAClsPredictor(
        configs=audio_args.configs,
        model_path=model_paths["音频分类模型"],
        use_gpu=audio_args.use_gpu
    )

    return {
        "vision": vision_models,
        "pollution": pollution_model,
        "audio": audio_predictor
    }


# ========================== 初始化系统 ==========================
models = load_all_models()
default_save_dir = Path(r'D:\ultralytics-main\zzzaaa_project\gui\test')


# ========================== 视觉处理函数 ==========================
def visual_detect(image_path, model_name, conf_threshold=0.5):
    """视觉检测处理"""
    try:
        if model_name == "污染分类模型":
            # 图像分类处理
            predicted_class, _ = classify_image(models["pollution"], image_path)
            label = pollution_levels.get(predicted_class, "未知污染类型")

            # 标注处理
            img = cv2.imread(image_path)
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype("simhei.ttf", 30)
            draw.text((10, 30), label, font=font, fill=(0, 255, 0))
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # 目标检测处理
            model = models["vision"][model_name]
            img_pil = detect_and_annotate(image_path, model, conf_threshold)
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 保存结果
        save_path = default_save_dir / Path(image_path).name
        default_save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)
        return str(save_path)

    except Exception as e:
        print(f"视觉处理错误: {str(e)}")
        return None


# ========================== 音频处理函数 ==========================
def audio_classify(audio_path, window_size=4.7, hop_length=2.7):
    """音频分类处理"""
    try:
        # 参数校验
        if window_size <= 0 or hop_length <= 0:
            raise ValueError("窗口参数必须大于0")
        if hop_length >= window_size:
            raise ValueError("步长应小于窗口尺寸")

        # 加载音频
        y, sr = librosa.load(audio_path, sr=16000)
        total_duration = len(y) / sr
        print(f"音频总时长: {total_duration:.2f}秒")

        # 分窗处理
        window_size_samples = int(window_size * sr)
        hop_length_samples = int(hop_length * sr)
        windows = []
        for i in range(0, len(y), hop_length_samples):
            window = y[i:i + window_size_samples]
            if len(window) / sr < 0.4:
                continue
            windows.append((i, window))

        # 执行预测
        results = []
        for idx, (start_sample, window) in enumerate(windows):
            start_time = start_sample / sr
            end_time = (start_sample + len(window)) / sr
            probabilities = models["audio"].mypredict(audio_data=window)
            results.append({
                "start": start_time,
                "end": end_time,
                "label": max(probabilities, key=probabilities.get),
                "confidence": probabilities[max(probabilities, key=probabilities.get)]
            })

        # 合并连续事件
        events = []
        current_event = None
        for res in results:
            if not current_event:
                current_event = res.copy()
            else:
                if res["label"] == current_event["label"] and res["start"] <= current_event["end"]:
                    current_event["end"] = res["end"]
                    current_event["confidence"] = max(current_event["confidence"], res["confidence"])
                else:
                    events.append(current_event)
                    current_event = res.copy()
        if current_event:
            events.append(current_event)

        # 生成报告
        report = [
            f"音频分析报告（总时长：{total_duration:.2f}秒）",
            "========================================"
        ]
        for idx, event in enumerate(events, 1):
            report.append(
                f"{idx}. 时间段：{event['start']:.2f}-{event['end']:.2f}秒 | "
                f"分类：{event['label']} | "
                f"置信度：{event['confidence']:.2%}"
            )
        return "\n".join(report)

    except Exception as e:
        print(f"音频处理错误: {str(e)}")
        return f"处理失败：{str(e)}"


# ========================== Gradio界面 ==========================
def create_interface():
    """创建多模态交互界面"""
    with gr.Blocks(title="智能检测平台", theme=gr.themes.Soft()) as demo:
        # ===== 视觉模块 =====
        with gr.Tab("🖼️ 视觉检测"):
            gr.Markdown("## 图像目标检测与污染分类")
            with gr.Row():
                with gr.Column(scale=1):
                    model_selector = gr.Dropdown(
                        label="选择视觉模型",
                        choices=["目标检测模型1", "室内检测模型", "VOC通用模型", "污染分类模型"],
                        value="目标检测模型1"
                    )
                    conf_slider = gr.Slider(0, 1, 0.5, label="检测置信度阈值")
                    img_upload = gr.Image(type="filepath", label="上传图像", height=300)
                    vis_btn = gr.Button("开始检测", variant="primary")

                with gr.Column(scale=2):
                    vis_output = gr.Image(label="检测结果", interactive=False, height=500)

        # ===== 音频模块 =====
        with gr.Tab("🎵 音频分析"):
            gr.Markdown("## 环境音频事件分析")
            with gr.Row():
                with gr.Column(scale=1):
                    audio_upload = gr.Audio(
                        type="filepath",
                        label="上传音频",
                        sources=["upload"],
                        waveform_options={"waveform_progress_color": "#36B6D5"}
                    )
                    with gr.Row():
                        win_size = gr.Number(4.7, label="分析窗口（秒）", minimum=0.5, maximum=30)
                        hop_size = gr.Number(2.7, label="滑动步长（秒）", minimum=0.1, maximum=15)
                    audio_btn = gr.Button("开始分析", variant="primary")

                with gr.Column(scale=2):
                    audio_output = gr.Textbox(
                        label="分析结果",
                        placeholder="检测结果将显示在此处...",
                        lines=15,
                        max_lines=20,
                        show_copy_button=True
                    )

        # ===== 事件绑定 =====
        vis_btn.click(
            fn=visual_detect,
            inputs=[img_upload, model_selector, conf_slider],
            outputs=vis_output
        )
        audio_btn.click(
            fn=audio_classify,
            inputs=[audio_upload, win_size, hop_size],
            outputs=audio_output
        )

    return demo


# ========================== 启动应用 ==========================
if __name__ == "__main__":
    app = create_interface()
    app.launch(
        # server_name="0.0.0.0",
        server_port=7861,
        share=True,
        favicon_path="favicon.ico"
    )