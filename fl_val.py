from ultralytics.models import YOLO
import os
import matplotlib.pyplot as plt

# 设置环境变量以避免库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



if __name__ == '__main__':
    # 加载模型
    model = YOLO(model=r'D:\ultralytics-main\zzzaaa_project\fl\model\best_830.pt')

    # 进行模型验证
    model.val(data=r'D:\ultralytics-main\zzzaaa_project\fl\fl_data.yaml', split='val', batch=1, device='0', project=r'D:\ultralytics-main\zzzaaa_project\fl\runs\val', name='fl',
              half=False)