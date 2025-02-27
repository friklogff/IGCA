import os
import json
import base64
import hashlib
import hmac
from datetime import datetime
from time import mktime
from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time
import websocket
import ssl
import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
from spkai.text_chat import chat
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ==================== 环境配置 ====================
load_dotenv()
SPARK_APP_ID = os.getenv("SPARKAI_APP_ID")
SPARK_API_KEY = os.getenv("SPARKAI_API_KEY")
SPARK_API_SECRET = os.getenv("SPARKAI_API_SECRET")
IMAGE_API_URL = "wss://spark-api.cn-huabei-1.xf-yun.com/v2.1/image"

print("✅ 环境变量加载完成")
print(f"🔑 APP_ID: {SPARK_APP_ID[:3]}...{SPARK_APP_ID[-3:]}")
print(f"🔑 API_KEY: {SPARK_API_KEY[:3]}...{SPARK_API_KEY[-3:]}")


# ==================== 数据结构定义 ====================
class DetectionResult:
    """环境检测结果数据结构"""

    def __init__(self, pollution_type, confidence, locations, severity_level):
        self.pollution_type = pollution_type  # 污染类型
        self.confidence = confidence  # 检测置信度
        self.locations = locations  # 污染区域坐标列表
        self.severity_level = severity_level  # 严重等级(1-5)

    def __str__(self):
        return (f"检测结果:\n"
                f"  类型: {self.pollution_type}\n"
                f"  置信度: {self.confidence:.2%}\n"
                f"  区域数: {len(self.locations)}\n"
                f"  严重等级: {self.severity_level}/5")


# ==================== 图像检测模块 ====================
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
            # 验证模型有效性
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

            # 执行预测
            print("🖼️ 执行YOLO预测...")
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


# ==================== PDF报告生成模块 ====================
class ReportGenerator:
    """PDF报告生成器"""

    @classmethod
    def create_pdf(cls, detection, analysis, output_path="环境报告.pdf"):
        """生成专业PDF报告"""
        print(f"\n🖨️ 开始生成PDF报告: {output_path}")
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()

        # 注册中文字体（使用 SimHei）
        font_path = "SimHei.ttf"  # 确保字体文件存在
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"字体文件未找到: {font_path}")

        pdfmetrics.registerFont(TTFont("SimHei", font_path))
        print("✅ 字体注册成功: SimHei")
        styles.add(ParagraphStyle(name="ChineseStyle", fontName="SimHei", fontSize=12, leading=20, wordWrap="CJK"))

        elements = []

        # 标题部分
        elements.append(Paragraph("<b>环境监测分析报告</b>", styles["ChineseStyle"]))
        elements.append(Spacer(1, 12))

        # 检测摘要
        summary_data = [
            ["污染类型", detection.pollution_type],
            ["置信度", f"{detection.confidence:.2%}"],
            ["污染区域", str(len(detection.locations))],
            ["严重等级", f"等级 {detection.severity_level}/5"]
        ]
        elements.append(Paragraph("<b>检测摘要</b>", styles["ChineseStyle"]))
        elements.append(Table(summary_data, colWidths=[100, 400], style=[
            ('FONTNAME', (0, 0), (-1, -1), 'SimHei'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('LEADING', (0, 0), (-1, -1), 14)
        ]))

        # AI分析
        elements.append(Spacer(1, 24))
        elements.append(Paragraph("<b>智能分析</b>", styles["ChineseStyle"]))
        analysis_table = [
            ["空间分布", analysis["spatial_distribution"]],
            ["严重程度", analysis["severity_analysis"]],
            ["总体评估", analysis["overview"]]
        ]
        elements.append(Table(analysis_table, colWidths=[120, 380], style=[
            ('FONTNAME', (0, 0), (-1, -1), 'SimHei'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('LEADING', (0, 0), (-1, -1), 14)
        ]))

        # 建议措施
        elements.append(Spacer(1, 24))
        elements.append(Paragraph("<b>治理建议</b>", styles["ChineseStyle"]))
        recs = [[f"{i + 1}. {rec}"] for i, rec in enumerate(analysis["recommendations"])]
        elements.append(Table(recs, colWidths=[500], style=[
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 0), (-1, -1), 'SimHei'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('LEADING', (0, 0), (-1, -1), 12)
        ]))

        doc.build(elements)
        print(f"✅ PDF报告已生成: {os.path.abspath(output_path)}")
        return output_path


# ==================== 主流程 ====================
def main(image_path):
    """完整处理流程"""
    print("\n" + "=" * 50)
    print("🚀 启动环境监测分析系统")
    print("=" * 50)

    try:
        # 初始化模块
        print("\n🛠️ 初始化系统组件...")
        detector = EnvironmentalDetector(r"D:\ultralytics-main\zzzaaa_project\shinei-zh\model\best900.pt")

        # 执行检测
        print("\n" + "-" * 50)
        print("🖼️ 开始图像分析流程")
        detection = detector.analyze_image(image_path)

        # 生成报告
        print("\n" + "-" * 50)
        print("📊 开始AI报告生成流程")
        image_base64 = _encode_image(image_path)
        prompt = _build_prompt(detection, image_base64)
        print(f"📝 构造的提示词:\n{prompt[:200]}...")  # 打印部分提示词
        response = chat(prompt)
        analysis = _parse_response(response)
        print(f"📋 解析后的分析结果:\n{json.dumps(analysis, indent=2, ensure_ascii=False)}")

        # 生成PDF报告
        print("\n" + "-" * 50)
        print("🖨️ 开始生成PDF报告")
        ReportGenerator.create_pdf(detection, analysis)

        print("\n" + "=" * 50)
        print("✅ 处理完成！")
        return analysis

    except Exception as e:
        print("\n‼️" + "=" * 50)
        print(f"❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def _encode_image(image_path):
    """编码图片为Base64"""
    print(f"🖼️ 编码图片: {os.path.basename(image_path)}")
    try:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        print(f"✅ 图片编码完成，长度: {len(image_base64) // 1024}KB")
        return image_base64
    except Exception as e:
        raise RuntimeError(f"图片编码失败: {str(e)}")


def _build_prompt(data, image_base64):
    """构建结构化提示模板"""
    return f"""
    作为环境监测专家，请根据以下检测数据生成JSON报告：
    {{
        "污染类型": "{data.pollution_type}",
        "置信度": {data.confidence:.2f},
        "污染区域数": {len(data.locations)},
        "严重等级": {data.severity_level}/5
    }}
    要求包含以下字段：
    {{
        "overview": "总体情况摘要（不少于100字）",
        "severity_analysis": "严重程度技术分析（包含数据支撑）",
        "spatial_distribution": "空间分布特征（根据坐标分析分布规律）",
        "recommendations": ["具体治理建议1", "具体治理建议2", "具体治理建议3"]
    }}
    请确保响应为纯JSON格式，不要包含任何Markdown语法。
    """


def _parse_response(response):
    """解析API响应"""
    print("\n🔍 解析API响应...")
    try:
        print(f"📥 原始响应内容:\n{response}")  # 打印完整响应内容
        # 尝试直接解析
        parsed = json.loads(response)
        print("✅ 直接解析JSON成功")
        return parsed
    except json.JSONDecodeError:
        print("⚠️ 检测到非标准JSON，尝试提取...")
        try:
            json_str = response[response.find('{'):response.rfind('}') + 1]
            parsed = json.loads(json_str)
            print("✅ 提取后解析成功")
            return parsed
        except Exception as e:
            raise ValueError(f"❌ 响应解析失败: {str(e)}\n原始响应: {response}")
    except Exception as e:
        raise ValueError(f"❌ 解析时发生意外错误: {str(e)}")


if __name__ == "__main__":
    test_image = r"D:\ultralytics-main\zzzaaa_project\shinei-zh\DLLG_datasets\DLLG_YOLO\images\train\000038.jpg"
    print("\n" + "⭐" * 50)
    print(" 测试用例启动 ".center(50, "⭐"))
    print("⭐" * 50)
    main(test_image)