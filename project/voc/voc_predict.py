from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO(model=r'D:\ultralytics-main\zzzaaa_project\voc\best_60.pt') #加载刚刚训练完的模型
    # 进行推理
    model.predict(source=r'D:\ultralytics-main\zzzaaa_project\voc\yolo\output\images\test\H-Battery25.jpg',  # source是要推理的图片路径这里使用数据集提供的图片
                  save=True,  # 是否在推理结束后保存结果
                  show=True,  # 是否在推理结束后显示结果
                  project='zzzaaa_project\\voc\\runs',  # 结果的保存路径
                  )