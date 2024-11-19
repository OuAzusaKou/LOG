import pyaudio
import wave
import numpy as np
import keyboard
import threading
import time

class AudioHandler:
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.p = pyaudio.PyAudio()
        self.recording = False
        self.frames = []

    def record_audio_with_keyboard(self, output_filename="output.wav"):
        """通过键盘控制录音"""
        print("请按住空格键进行录音，松开结束录音...")
        
        # 创建音频流
        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        def on_press(event):
            if event.name == 'space' and not self.recording:
                self.recording = True
                self.frames = []
                print("\n开始录音...")

        def on_release(event):
            if event.name == 'space' and self.recording:
                self.recording = False
                print("录音结束")
                
                # 保存录音文件
                wf = wave.open(output_filename, 'wb')
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(self.frames))
                wf.close()
                
                # 自动播放录音
                self.play_audio(output_filename)

        # 注册键盘事件
        keyboard.on_press(on_press)
        keyboard.on_release(on_release)

        try:
            while True:
                if self.recording:
                    data = stream.read(self.CHUNK)
                    self.frames.append(data)
                time.sleep(0.01)  # 减少CPU使用率
        except KeyboardInterrupt:
            print("\n录音程序已退出")
        finally:
            stream.stop_stream()
            stream.close()
            keyboard.unhook_all()

    def play_audio(self, filename):
        """播放音频文件"""
        wf = wave.open(filename, 'rb')
        
        stream = self.p.open(
            format=self.p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )

        print("开始播放...")
        data = wf.readframes(self.CHUNK)

        while data:
            stream.write(data)
            data = wf.readframes(self.CHUNK)

        print("播放结束")

        stream.stop_stream()
        stream.close()

    def __del__(self):
        self.p.terminate()

# 使用示例
if __name__ == "__main__":
    audio_handler = AudioHandler()
    
    # 启动键盘控制的录音功能
    audio_handler.record_audio_with_keyboard()
