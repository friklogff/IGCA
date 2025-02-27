image_understanding.py
Python
复制
import os
import json
import base64
import hashlib
import hmac
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time
import websocket
from PIL import Image
import io
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
appid = os.environ.get("SPARKAI_APP_ID")
api_secret = os.environ.get("SPARKAI_API_SECRET")
api_key = os.environ.get("SPARKAI_API_KEY")
image_url = "wss://spark-api.cn-huabei-1.xf-yun.com/v2.1/image"

def generate_signature(url):
    """
    生成 WebSocket 请求所需的签名。
    输入：
        url: WebSocket 的 URL
    输出：
        signature: 签名字符串
    """
    now = datetime.now()
    date = format_date_time(mktime(now.timetuple()))
    signature_origin = f"host: {urlparse(url).netloc}\n"
    signature_origin += f"date: {date}\n"
    signature_origin += f"GET {urlparse(url).path} HTTP/1.1"
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'), digestmod=hashlib.sha256).digest()
    signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')
    authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    return urlencode({"authorization": authorization, "date": date, "host": urlparse(url).netloc})

def analyze_image(image_path, question):
    """
    对图像进行理解并回答问题。
    输入：
        image_path: 图像文件的路径 (str)
        question: 针对图像的问题 (str)
    输出：
        answer: 图像理解的结果 (str)
    """
    answer = ""  # 在函数内部定义 answer

    # 将图像转换为 Base64 格式
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # WebSocket 回调函数
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
            nonlocal answer  # 使用 nonlocal 以修改外部变量
            answer += content
            if status == 2:
                ws.close()

    def on_error(ws, error):
        print("### error:", error)

    def on_close(ws, one, two):
        print(" ")

    def on_open(ws):
        data = json.dumps({
            "header": {"app_id": appid},
            "parameter": {"chat": {"domain": "imagev3", "temperature": 0.5, "top_k": 4, "max_tokens": 2028}},
            "payload": {
                "message": {
                    "text": [
                        {"role": "user", "content": image_base64, "content_type": "image"},
                        {"role": "user", "content": question, "content_type": "text"}
                    ]
                }
            }
        })
        ws.send(data)

    # 启动 WebSocket
    websocket.enableTrace(False)
    url = image_url + '?' + generate_signature(image_url)
    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    return answer
text_chat.py
Python
复制
import os
import json
import base64
import hashlib
import hmac
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time
import websocket
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
appid = os.environ.get("SPARKAI_APP_ID")
api_secret = os.environ.get("SPARKAI_API_SECRET")
api_key = os.environ.get("SPARKAI_API_KEY")
Spark_url = "wss://spark-api.xf-yun.com/v4.0/chat"
domain = "4.0Ultra"

def generate_signature(url):
    """
    生成 WebSocket 请求所需的签名。
    输入：
        url: WebSocket 的 URL
    输出：
        signature: 签名字符串
    """
    now = datetime.now()
    date = format_date_time(mktime(now.timetuple()))
    signature_origin = f"host: {urlparse(url).netloc}\n"
    signature_origin += f"date: {date}\n"
    signature_origin += f"GET {urlparse(url).path} HTTP/1.1"
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'), digestmod=hashlib.sha256).digest()
    signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')
    authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    return urlencode({"authorization": authorization, "date": date, "host": urlparse(url).netloc})

def chat(message):
    """
    与星火 AI 进行文本聊天。
    输入：
        message: 用户输入的文本消息 (str)
    输出：
        response: AI 的回复 (str)
    """
    response = ""  # 在函数内部定义 response

    # WebSocket 回调函数
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
            nonlocal response  # 使用 nonlocal 以修改外部变量
            response += content
            if status == 2:
                ws.close()

    def on_error(ws, error):
        print("### error:", error)

    def on_close(ws, one, two):
        print(" ")

    def on_open(ws):
        data = json.dumps({
            "header": {"app_id": appid},
            "parameter": {"chat": {"domain": domain, "temperature": 0.5, "top_k": 4, "max_tokens": 2028}},
            "payload": {"message": {"text": [{"role": "user", "content": message}]}}
        })
        ws.send(data)

    # 启动 WebSocket
    websocket.enableTrace(False)
    url = Spark_url + '?' + generate_signature(Spark_url)
    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    return response
main.py
Python
复制
from image_understanding import analyze_image
from text_chat import chat

def main():
    # 图像理解示例
    image_path = r"D:\ultralytics-main\zzzaaa_project\shinei-zh\DLLG_datasets\DLLG_YOLO\images\train\000038.jpg"
    question = "What is in the image?"
    answer = analyze_image(image_path, question)
    print(f"Image Understanding Answer: {answer}")

    # 文本聊天示例
    message = "Hello, how are you?"
    response = chat(message)
    print(f"Chat Response: {response}")

if __name__ == "__main__":
    main()