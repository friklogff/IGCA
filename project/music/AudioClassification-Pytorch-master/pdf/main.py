from utils import environment_detector
if __name__ == "__main__":
    test_image = r"jd1.jpg"
    print("\n" + "⭐" * 50)
    print(" 测试用例启动 ".center(50, "⭐"))
    print("⭐" * 50)
    environment_detector(test_image)
