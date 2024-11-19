import keyboard
import time
from datetime import datetime
import os
import pyscreenshot as ImageGrab
from Xvfb import Xvfb

# 创建虚拟显示
vdisplay = Xvfb()
vdisplay.start()

# 定义要截图的区域（左上角x, 左上角y, 右下角x, 右下角y）
SCREEN_AREA = (100, 100, 300, 300)  # 这些数值需要根据实际需求调整

# 创建保存截图的文件夹
SCREENSHOTS_DIR = "screenshots"
if not os.path.exists(SCREENSHOTS_DIR):
    os.makedirs(SCREENSHOTS_DIR)

def capture_and_save():
    try:
        # 截取指定区域的屏幕
        screenshot = ImageGrab.grab(bbox=SCREEN_AREA)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SCREENSHOTS_DIR, f"screenshot_{timestamp}.png")
        
        # 保存截图
        screenshot.save(filename)
        print(f"截图已保存: {filename}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

def main():
    print("程序已启动...")
    print("按下 'F9' 键进行截图")
    print("按下 'Esc' 键退出程序")
    
    # 注册热键
    keyboard.on_press_key('F9', lambda _: capture_and_save())
    keyboard.wait('esc')  # 等待按下 Esc 键退出

    # 程序结束时关闭虚拟显示
    vdisplay.stop()

if __name__ == "__main__":
    main()
