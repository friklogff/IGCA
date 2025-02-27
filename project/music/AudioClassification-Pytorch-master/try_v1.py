import pdfkit
import os

class ReportGenerator:
    @staticmethod
    def create_pdf(detection_result, analysis_report, output_path="report.pdf"):
        """
        根据检测结果和分析报告生成PDF文件
        :param detection_result: 检测结果对象
        :param analysis_report: 分析报告内容（字符串）
        :param output_path: 输出PDF文件路径
        """
        html_content = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h1 {{ color: #333; }}
                p {{ font-size: 14px; }}
            </style>
        </head>
        <body>
            <h1>环境监测报告</h1>
            <p><strong>检测结果：</strong>{detection_result.pollution_type}</p>
            <p><strong>置信度：</strong>{detection_result.confidence:.2f}</p>
            <p><strong>严重等级：</strong>{detection_result.severity_level}</p>
            <p><strong>AI分析报告：</strong></p>
            <p>{analysis_report}</p>
        </body>
        </html>
        """
        path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
        pdfkit.from_string(html_content, output_path, configuration=config)
        return output_path

def main_workflow(image_path: str):
    try:
        report_path = ReportGenerator.create_pdf(detection_result, analysis_report)
        print(f"\n✅ 报告生成成功：{os.path.abspath(report_path)}")
        return report_path
    except Exception as e:
        print(f"\n❌ 处理失败：{str(e)}")
        return None

if __name__ == "__main__":
    test_image = r"D:\ultralytics-main\zzzaaa_project\shinei-zh\DLLG_datasets\DLLG_YOLO\images\train\000038.jpg"
    report_path = main_workflow(test_image)