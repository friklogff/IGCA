import gradio as gr

def process_video(video):
    # 视频处理逻辑
    return video

video_interface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload a video file"),
    outputs="video"
)
video_interface.launch()