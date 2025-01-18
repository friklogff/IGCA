import gradio as gr

def operation(choice1, choice2):
    re = "您还未选择哦~"  # 默认返回值
    if choice1 is not None and choice2 is not None:
        re = choice1 + "，" + choice2
    else:
        gr.Warning("还有选项没选哦~")
    return re

with gr.Blocks(css="css/index.css") as demo:
    gr.Markdown(value="<h1 align='center' style='font:darkcyan;' class='title'>一级标题</h1>")
    gr.Markdown(value="<hr>")  # 分割线
    gr.Markdown(value="<br>")  # 换行
    gr.Markdown(value="<br>")  # 换行

    with gr.Row():
        with gr.Column():
            dp1 = gr.Dropdown(label="性别", choices=["男", "女"], elem_classes="dropDown sex")
        #     elem_classes="dropDown sex"重命名
        with gr.Column():
            dp2 = gr.Dropdown(label="特长", choices=["腿", "胳膊"], elem_id="dropDownSpecialty")

    gr.Markdown(value="<br>")  # 换行
    gr.Markdown(value="<br>")  # 换行
    gr.Markdown(value="<hr>")  # 分割线

    btn = gr.Button(value="获取下拉框所选值")
    outputText = gr.Textbox(label="所选值", lines=3, placeholder="下拉框所选值")
    btn.click(fn=operation, inputs=[dp1, dp2], outputs=[outputText])

# 启动 Gradio 应用程序
demo.launch()