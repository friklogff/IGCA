# # # Gradio_App.py
# #
# # import gradio as gr
# #
# # def greet(name):
# #     return f"Hello {name}! Welcome to 2024!"
# #
# # iface = gr.Interface(fn=greet, inputs="text", outputs="text")
# # iface.launch()
# import json
# import base64
# import requests
#
# # HAI服务器IP地址
# your_ip = '127.0.0.1'
# # SD api 监听的端口
# your_port = 7860
#
# def submit_post(url: str, data: dict):
#     """
#     提交POST请求到给定URL，并携带给定数据。
#     """
#     return requests.post(url, data=json.dumps(data))
#
# def save_encoded_image(b64_image: str, output_path: str):
#     """
#     将给定的Base64编码图像保存到指定输出路径。
#     """
#     with open(output_path, "wb") as image_file:
#         image_file.write(base64.b64decode(b64_image))
#
# if __name__ == '__main__':
#     # /sdapi/v1/txt2img
#     txt2img_url = f'http://{your_ip}:{your_port}/sdapi/v1/txt2img'
#     data = {
#         'prompt': 'a pretty cat,cyberpunk art,kerem beyit,verycute robot zen,Playful,Independent,beeple |',
#         'negative_prompt': '(deformed,distorted,disfigured:1.0),poorlydrawn,bad anatomy,wrong anatomy,extra limb,missing limb,floating limbs,(mutatedhands and fingers:1.5),disconnectedlimbs,mutation,mutated,ugly,disgusting,blurry,amputation,flowers,human,man,woman',
#         'Steps': 50,
#         'Seed': 1791574510
#     }
#     response = submit_post(txt2img_url, data)
#     save_encoded_image(response.json()['images'][0], 'cat.png')
import json
import base64
import requests
import gradio as gr

# HAI服务器IP地址
your_ip = '127.0.0.1'
# SD api 监听的端口
your_port = 7860


def submit_post(url: str, data: dict):
    return requests.post(url, data=json.dumps(data))


def save_encoded_image(b64_image: str, output_path: str):
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))


def generate_image(prompt, negative_prompt, steps, seed):
    txt2img_url = f'http://{your_ip}:{your_port}/sdapi/v1/txt2img'
    data = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'Steps': steps,
        'Seed': seed
    }
    response = submit_post(txt2img_url, data)
    save_encoded_image(response.json()['images'][0], 'cat.png')
    return 'cat.png'


with gr.Tab("Text to Image Generation"):
    prompt = gr.Textbox(lines=5, label="Prompt")
    neg_prompt = gr.Textbox(label="Negative Prompt")
    steps = gr.Number(label="Steps")
    seed = gr.Number(label="Seed")
    generated_image = gr.Image(type="pil", label="Generated Image")

iface = gr.Interface(fn=generate_image, inputs=[prompt, neg_prompt, steps, seed], outputs=generated_image)
iface.launch()
