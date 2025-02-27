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
import os
import SparkApi
import base64
import datetime
import hashlib
import hmac
import json
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket
import _thread as thread
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

appid = os.environ.get("SPARKAI_APP_ID")
api_secret = os.environ.get("SPARKAI_API_SECRET")
api_key = os.environ.get("SPARKAI_API_KEY")

domain = "4.0Ultra"
Spark_url = "wss://spark-api.xf-yun.com/v4.0/chat"
image_url = "wss://spark-api.cn-huabei-1.xf-yun.com/v2.1/image"

text = []
connection_status = {"connected": False}

def getText(role, content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text

def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length

def checklen(text):
    while getlength(text) > 8000:
        del text[0]
    return text

def generate_signature(url):
    now = datetime.now()
    date = format_date_time(mktime(now.timetuple()))
    signature_origin = f"host: {urlparse(url).netloc}\n"
    signature_origin += f"date: {date}\n"
    signature_origin += f"GET {urlparse(url).path} HTTP/1.1"
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'), digestmod=hashlib.sha256).digest()
    signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')
    authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    v = {
        "authorization": authorization,
        "date": date,
        "host": urlparse(url).netloc
    }
    return urlencode(v)

def on_message(ws, message):
    data = json.loads(message)
    code = data['header']['code']
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        print(content, end="")
        global answer
        answer += content
        if status == 2:
            ws.close()

def on_error(ws, error):
    print("### error:", error)

def on_close(ws, one, two):
    print(" ")

def on_open(ws):
    thread.start_new_thread(run, (ws,))

def run(ws, *args):
    data = json.dumps({
        "header": {
            "app_id": appid
        },
        "parameter": {
            "chat": {
                "domain": "imagev3",
                "temperature": 0.5,
                "top_k": 4,
                "max_tokens": 2028
            }
        },
        "payload": {
            "message": {
                "text": [
                    {
                        "role": "user",
                        "content": image_base64,
                        "content_type": "image"
                    },
                    {
                        "role": "user",
                        "content": Input,
                        "content_type": "text"
                    }
                ]
            }
        }
    })
    ws.send(data)

def analyze_image(image, question):
    global image_base64, Input, answer
    pil_image = Image.fromarray(image)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    Input = question
    answer = ""
    print("答:", end="")
    websocket.enableTrace(False)
    url = image_url + '?' + generate_signature(image_url)
    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = appid
    ws.question = question
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    return answer

def chat(message):
    question = checklen(getText("user", message))
    SparkApi.answer = ""
    print("星火:", end="")
    SparkApi.main(appid, api_key, api_secret, Spark_url, domain, question)
    getText("assistant", SparkApi.answer)
    return SparkApi.answer

class Ws_Param:
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text

        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {
            "aue": "lame",
            "auf": "audio/L16;rate=16000",
            "vcn": "xiaoyan",
            "tte": "utf8"
        }
        self.Data = {
            "status": 2,
            "text": str(base64.b64encode(self.Text.encode('utf-8')), "UTF8")
        }

    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 生成签名
        signature_origin = f"host: ws-api.xfyun.cn\ndate: {date}\nGET /v2/tts HTTP/1.1"
        signature_sha = hmac.new(
            self.APISecret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode('utf-8')

        # 构造鉴权头
        authorization = (
            f'api_key="{self.APIKey}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature_sha}"'
        )
        authorization = base64.b64encode(authorization.encode('utf-8')).decode('utf-8')

        # 生成最终 URL
        params = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        return f"{url}?{urlencode(params)}"

def on_message_tts(ws, message):
    try:
        msg = json.loads(message)
        if msg["code"] != 0:
            print(f"错误: {msg['message']}")
            return

        audio = base64.b64decode(msg["data"]["audio"])
        with open('output.mp3', 'ab') as f:
            f.write(audio)

        if msg["data"]["status"] == 2:
            print("转换成功！文件已保存为 output.mp3")
            ws.close()  # 关闭连接

    except Exception as e:
        print(f"处理消息失败: {str(e)}")

def on_error_tts(ws, error):
    if connection_status["connected"]:
        print(f"网络错误: {str(error)}")

def on_close_tts(ws, *args):
    print("### 连接已关闭 ###")
    connection_status["connected"] = False

def on_open_tts(ws):
    def run():
        connection_status["connected"] = True
        data = {
            "common": wsParam.CommonArgs,
            "business": wsParam.BusinessArgs,
            "data": wsParam.Data
        }
        ws.send(json.dumps(data))
        # 清空旧文件
        if os.path.exists('output.mp3'):
            os.remove('output.mp3')
    thread.start_new_thread(run, ())

def text_to_speech(text):
    global wsParam
    wsParam = Ws_Param(
        APPID=appid,
        APIKey=api_key,
        APISecret=api_secret,
        Text=text
    )

    ws = websocket.WebSocketApp(
        wsParam.create_url(),
        on_open=on_open_tts,
        on_message=on_message_tts,
        on_error=on_error_tts,
        on_close=on_close_tts
    )

    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    return "output.mp3"

# ========================== 模型配置 ==========================
model_paths = {
    # 视觉模型
    "垃圾分类模型": r'D:\ultralytics-main\zzzaaa_project\fl\model\best520.pt',
    "室内检测模型": r'D:\ultralytics-main\zzzaaa_project\shinei-zh\model\best900.pt',
    "常见分类模型": r'D:\ultralytics-main\zzzaaa_project\voc\voc_best_220.pt',
    "水下分类模型": r'D:\ultralytics-main\zzzaaa_project\sx\weights\best_sx.pt',
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
default_save_dir = Path(r'E:\ultralytics-main\zzzaaa_project\gui\test')

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
    custom_css = """
    .title-block {
        background: linear-gradient(135deg, #36B6D5, #2E8B57);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
    }
    .title-block h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5em;
    }
    .title-block p {
        color: rgba(255,255,255,0.9) !important;
        margin: 10px 0 0 0;
    }
    .section-title {
        border-left: 4px solid #36B6D5;
        padding-left: 12px;
        margin: 15px 0;
        color: #2E8B57;
    }
    .footer {
        text-align: center;
        padding: 15px;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 25px;
    }
    """

    with gr.Blocks(title="智能检测平台", theme=gr.themes.Soft(), css=custom_css) as demo:
        # ===== 头部横幅 =====
        gr.HTML("""
        <div class="title-block">
            <h1>智能多模态分析平台</h1>
            <p>视觉检测 × 音频分析 × 智能识别</p>
        </div>
        """)

        # ===== 视觉检测模块 =====
        with gr.Tab("🖼️ 视觉检测", id="vision_tab"):
            with gr.Accordion("📌 使用指南", open=False):
                gr.Markdown("""
                ### 视觉检测功能说明：
                1. 支持 **4种检测模式**：目标检测/污染分类
                2. 支持 **JPEG/PNG** 格式图片
                3. 实时显示处理耗时和置信度
                """)
            with gr.Row():
                # 左侧控制面板
                with gr.Column(scale=1):
                    gr.Markdown("### 模型配置", elem_classes="section-title")
                    model_selector = gr.Dropdown(
                        label="选择视觉模型",
                        choices=["垃圾分类模型", "室内检测模型", "常见分类模型", "水下分类模型", "污染分类模型"],
                        value="垃圾分类模型"
                    )
                    conf_slider = gr.Slider(0, 1, 0.5, label="检测置信度阈值")

                    gr.Markdown("### 输入源", elem_classes="section-title")
                    img_upload = gr.Image(
                        type="filepath",
                        label="上传图像",
                        height=300
                    )
                    vis_btn = gr.Button("开始检测", variant="primary")

                # 右侧结果展示
                with gr.Column(scale=2):
                    vis_output = gr.Image(
                        label="检测结果预览",
                        interactive=False,
                        height=500
                    )

        # ===== 音频分析模块 =====
        with gr.Tab("🎵 音频分析", id="audio_tab"):
            with gr.Accordion("📌 使用指南", open=False):
                gr.Markdown("""
                ### 音频分析功能说明：
                1. 支持 **WAV** 格式音频文件
                2. 分析音频事件并生成报告
                3. 调整窗口大小和滑动步长以优化分析
                """)

            with gr.Row():
                # 左侧控制区
                with gr.Column(scale=1):
                    gr.Markdown("### 音频输入", elem_classes="section-title")
                    audio_upload = gr.Audio(
                        type="filepath",
                        label="上传音频文件",
                        sources=["upload"],
                        waveform_options={"waveform_progress_color": "#36B6D5"}
                    )

                    gr.Markdown("### 分析参数", elem_classes="section-title")
                    with gr.Row():
                        win_size = gr.Number(4.7, label="窗口大小（秒）", precision=1)
                        hop_size = gr.Number(2.7, label="滑动步长（秒）", precision=1)
                    audio_btn = gr.Button("开始分析", variant="primary")

                # 右侧结果区
                with gr.Column(scale=2):
                    audio_output = gr.Textbox(
                        label="分析结果报告",
                        placeholder="等待分析结果...",
                        lines=15,
                        show_copy_button=True
                    )

        # ===== 图像理解模块 =====
        with gr.Tab("🖼️ 图像理解", id="image_understanding_tab"):
            with gr.Accordion("📌 使用指南", open=False):
                gr.Markdown("""
                ### 图像理解功能说明：
                1. 上传图像并输入问题
                2. 获取图像的智能分析结果
                """)

            gr.Markdown("# Image Understanding with SparkAI")
            image_input = gr.Image(type="numpy", label="Upload Image")
            question_input = gr.Textbox(label="Question")
            image_output = gr.Textbox(label="Answer")
            image_button = gr.Button("Analyze Image")
            image_button.click(analyze_image, inputs=[image_input, question_input], outputs=image_output)

        # ===== 聊天模块 =====x
        with gr.Tab("💬 聊天", id="chat_tab"):
            with gr.Accordion("📌 使用指南", open=False):
                gr.Markdown("""
                ### 聊天功能说明：
                1. 输入文本与AI进行对话
                2. 获取智能回复
                """)

            gr.Markdown("# Chat with SparkAI")
            chat_input = gr.Textbox(label="Your Message")
            chat_output = gr.Textbox(label="Response")
            chat_button = gr.Button("Send")
            chat_button.click(chat, inputs=chat_input, outputs=chat_output)

        # ===== 文本转语音模块 =====
        with gr.Tab("🗣️ 文本转语音", id="text_to_speech_tab"):
            with gr.Accordion("📌 使用指南", open=False):
                gr.Markdown("""
                ### 文本转语音功能说明：
                1. 输入文本内容
                2. 生成并下载语音文件
                """)

            gr.Markdown("# Text to Speech with SparkAI")
            tts_input = gr.Textbox(label="Enter Text")
            tts_output = gr.File(label="Download Audio")
            tts_button = gr.Button("Convert to Speech")
            tts_button.click(text_to_speech, inputs=tts_input, outputs=tts_output)

        # ===== 页脚 =====
        gr.HTML("""
        <div class="footer">
            <p>© 2024 智能分析平台 | 技术支持: AI Lab | 版本: 2.1.0</p>
            <div style="display: flex; justify-content: center; gap: 15px;">
                <a href="#" style="color: #36B6D5;">用户协议</a>
                <a href="#" style="color: #36B6D5;">隐私政策</a>
                <a href="#" style="color: #36B6D5;">问题反馈</a>
            </div>
        </div>
        """)

        # ===== 事件绑定 =====
        vis_btn.click(
            visual_detect,
            inputs=[img_upload, model_selector, conf_slider],
            outputs=vis_output
        )
        audio_btn.click(
            audio_classify,
            inputs=[audio_upload, win_size, hop_size],
            outputs=audio_output
        )

    return demo

# ========================== 启动应用 ==========================
if __name__ == "__main__":
    app = create_interface()
    app.launch(
        # server_name="0.0.0.0",
        server_port=7860,
        share=True,
        favicon_path="favicon.ico"
    )
