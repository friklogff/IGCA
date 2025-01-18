from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('./ultralytics/cfg/models/11/yolo11.yaml')
    model.load('./yolo11n.pt')
    results = model.train(
        data='./data.yaml',
        epochs=5,
        imgsz=640,
        cache=False,
        batch=16,
        device='0',
        single_cls=False,
        amp=True
        )