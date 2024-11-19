import cv2
import numpy as np
from typing import Tuple, Optional

def read_video_segment(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    从视频文件中读取指定时间段的片段
    
    参数:
        video_path (str): 视频文件的路径
        start_time (float, optional): 开始时间（秒），如果为 None 则从视频开始处读取
        end_time (float, optional): 结束时间（秒），如果为 None 则读取到视频结束
        
    返回:
        Tuple[np.ndarray, float]: 包含视频帧的数组和视频的 FPS
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    
    # 如果没有指定时间参数，直接读取整个视频
    if start_time is None and end_time is None:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError("未能读取到任何帧")
        return np.array(frames), fps
    
    # 处理时间参数
    start_time = 0.0 if start_time is None else start_time
    start_frame = int(start_time * fps)
    end_frame = total_frames if end_time is None else min(int(end_time * fps), total_frames)
    
    # 检查时间参数是否有效
    if start_frame >= end_frame:
        raise ValueError("开始时间必须小于结束时间")
    if start_time < 0 or (end_time and end_time > video_duration):
        raise ValueError("时间范围超出视频长度")
    
    # 设置视频读取位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 读取指定范围的帧
    frames = []
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    # 释放视频对象
    cap.release()
    
    if not frames:
        raise ValueError("未能读取到任何帧")
    
    return np.array(frames), fps

def save_video_mp4(
    frames: np.ndarray,
    output_path: str,
    fps: float,
    codec: str = 'mp4v'
) -> None:
    """
    将帧数组保存为 MP4 视频文件
    
    参数:
        frames (np.ndarray): 视频帧数组，形状为 (n_frames, height, width, channels)
        output_path (str): 输出视频文件的路径（应以 .mp4 结尾）
        fps (float): 视频的帧率
        codec (str): 视频编码器，默认使用 'mp4v'
    """
    if not frames.size:
        raise ValueError("帧数组为空")
    
    if not output_path.endswith('.mp4'):
        output_path += '.mp4'
    
    # 获取视频尺寸
    height, width = frames[0].shape[:2]
    
    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError("无法创建输出视频文件")
    
    try:
        # 写入每一帧
        for frame in frames:
            out.write(frame)
    finally:
        # 确保释放资源
        out.release()

# 使用示例更新
if __name__ == "__main__":
    try:
        # 读取视频片段
        frames, fps = read_video_segment(
            video_path="./The Wandering Earth 2.mp4",
            start_time=2.0,
            end_time=90
        )
        print(f"成功读取视频片段，共 {len(frames)} 帧，FPS: {fps}")
        
        # 保存视频片段
        save_video_mp4(
            frames=frames,
            output_path="output_video.mp4",
            fps=fps
        )
        print("视频已成功保存")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
