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