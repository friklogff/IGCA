import os
import json
import base64

import numpy as np

from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet
from model_utils import load_yolo_models, detect_and_annotate, load_classification_model, classify_image

# 注册中文字体
font_path = r"C:\Windows\Fonts\simsun.ttc"
pdfmetrics.registerFont(TTFont('SimSun', font_path))

class DetectionResult:
    """环境检测结果数据结构"""
    def __init__(self, pollution_type, confidence, locations, severity_level):
        self.pollution_type = pollution_type
        self.confidence = confidence
        self.locations = locations
        self.severity_level = severity_level

    def __str__(self):
        return (f"检测结果:\n"
                f"  类型: {self.pollution_type}\n"
                f"  置信度: {self.confidence:.2%}\n"
                f"  区域数: {len(self.locations)}\n"
                f"  严重等级: {self.severity_level}/5")

class EnvironmentalDetector:
    """基于YOLO的环境污染检测器"""
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        self.class_names = self.model.names if hasattr(self.model, 'names') else []
        print(f"📚 类别标签加载完成，共{len(self.class_names)}个类别")

    def _load_model(self, model_path):
        """加载预训练模型"""
        print(f"\n🔄 正在加载模型: {os.path.basename(model_path)}")
        try:
            model = YOLO(model_path)
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(test_img, verbose=False)
            print(f"✅ 模型验证通过，输入尺寸: {model.overrides['imgsz']}")
            return model
        except Exception as e:
            raise RuntimeError(f"❌ 模型加载失败: {str(e)}")

    def analyze_image(self, image_path):
        """执行多维度环境分析"""
        print(f"\n🔍 开始分析图像: {os.path.basename(image_path)}")
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像不存在: {image_path}")

            results = self.model.predict(
                source=image_path,
                conf=0.5,
                save=False,
                verbose=False
            )

            if not results or len(results[0].boxes) == 0:
                print("⚠️ 未检测到污染目标")
                return DetectionResult("未检测到污染", 0.0, [], 0)

            boxes = results[0].boxes
            max_conf_idx = boxes.conf.argmax()
            main_class = int(boxes.cls[max_conf_idx])

            result = DetectionResult(
                pollution_type=self.class_names.get(main_class, "未知污染"),
                confidence=float(boxes.conf[max_conf_idx]),
                locations=[tuple(map(int, box.xyxy[0].cpu().numpy())) for box in boxes],
                severity_level=self._calculate_severity(boxes)
            )
            print("🎯 检测完成")
            print(result)
            return result
        except Exception as e:
            print(f"❌ 分析错误: {str(e)}")
            return DetectionResult("分析失败", 0.0, [], 0)

    def _calculate_severity(self, boxes):
        """计算污染严重等级"""
        count = len(boxes)
        avg_conf = boxes.conf.mean().item()
        level = min(count + int(avg_conf * 10), 5)
        print(f"📊 严重等级计算: 目标数={count}, 平均置信度={avg_conf:.2f} → 等级{level}")
        return level

def analyze_image(image_path):
    # 加载模型
    models = {
        "垃圾分类模型": r"D:\ultralytics-main\zzzaaa_project\fl\model\best520.pt",
        "常见分类模型": r"D:\ultralytics-main\zzzaaa_project\voc\voc_best_220.pt",
        "室内检测模型": r"D:\ultralytics-main\zzzaaa_project\shinei-zh\model\best900.pt",
        "河道分类模型": r"D:\ultralytics-main\zzzaaa_project\hd\model\best_model\75.pdparams",
        "水下分类模型": r"D:\ultralytics-main\zzzaaa_project\sx\weights\best_sx.pt"
    }

    # 检测图像
    results = {}
    for name, model_path in models.items():
        if name != "河道分类模型":  # 河道分类模型是分类模型，不是YOLO模型
            detector = EnvironmentalDetector(model_path)
            detection_result = detector.analyze_image(image_path)
            results[name] = detection_result

    # 分类图像
    classification_model = load_classification_model(models["河道分类模型"])
    predicted_class, confidence = classify_image(classification_model, image_path)
    results["河道分类模型"] = {"class": predicted_class, "confidence": confidence}

    return results


def generate_report(analysis, image_path):
    pdf_path = image_path.replace(".jpg", "_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # 添加标题
    c.setFont("SimSun", 16)
    c.drawCentredString(width / 2, height - 50, "综合环境监测分析报告")

    # 添加分析结果
    styles = getSampleStyleSheet()
    style = styles["BodyText"]
    style.fontName = "SimSun"
    style.fontSize = 12
    story = []

    for model, result in analysis.items():
        story.append(Paragraph(f"{model} 检测结果:", style))
        story.append(Paragraph(str(result), style))

    frame = Frame(50, 50, width - 100, height - 150, showBoundary=1)
    frame.addFromList(story, c)

    c.save()
    return pdf_path

def main(image_path):
    analysis = analyze_image(image_path)
    pdf_report = generate_report(analysis, image_path)
    print(f"报告已生成: {pdf_report}")

if __name__ == "__main__":
    test_image = r"D:\ultralytics-main\zzzaaa_project\shinei-zh\DLLG_datasets\DLLG_YOLO\images\train\000038.jpg"
    main(test_image)
