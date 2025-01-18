import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO(r'D:\ultralytics-main\zzzaaa_project\fl\model\best520.pt')  # 替换为你的模型文件路径



# 打开摄像头
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头

while True:
    # 读取摄像头的帧
    ret, frame = cap.read()
    if not ret:
        break

    # 进行预测
    results = model.predict(frame)

    # 绘制检测框和标签
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_id = int(box.cls.item())  # 从张量中提取类别索引
            conf = box.conf.item()  # 从张量中提取置信度
            label = f'{model.names[class_id]}: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示帧
    cv2.imshow('YOLO Real-time Detection', frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()