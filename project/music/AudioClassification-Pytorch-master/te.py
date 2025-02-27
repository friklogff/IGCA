import gradio as gr
from transformers import pipeline

# 加载语音识别模型
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# 语音识别函数
def recognize_speech(audio):
    if audio is None:
        return "未检测到音频输入，请说话！"
    text = asr(audio)["text"]
    return text

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# 按住说话示例")
    audio_input = gr.Audio(label="按住说话", sources=["microphone"], type="filepath")
    text_output = gr.Textbox(label="识别结果")
    audio_input.change(recognize_speech, inputs=audio_input, outputs=text_output)

demo.launch()