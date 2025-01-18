from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO(model=r'D:\ultralytics-main\runs\detect\train16\weights\best.pt') #加载刚刚训练完的模型
    # 进行推理
    model.predict(source=r'111.jpg',  # source是要推理的图片路径这里使用数据集提供的图片
                  save=True,  # 是否在推理结束后保存结果
                  show=True,  # 是否在推理结束后显示结果
                  project='runs/detect',  # 结果的保存路径
                  )