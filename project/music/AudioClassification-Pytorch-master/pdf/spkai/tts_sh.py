import os
import base64
import datetime
import hashlib
import hmac
import json
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket
import _thread as thread
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