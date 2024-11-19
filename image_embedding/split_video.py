import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional

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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    
    start_time = 0.0 if start_time is None else start_time
    start_frame = int(start_time * fps)
    end_frame = total_frames if end_time is None else min(int(end_time * fps), total_frames)
    
    if start_frame >= end_frame:
        raise ValueError("开始时间必须小于结束时间")
    if start_time < 0 or (end_time and end_time > video_duration):
        raise ValueError("时间范围超出视频长度")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if not frames:
        raise ValueError("未能读取到任何帧")
    
    return np.array(frames), fps

def detect_scene_changes(frames: np.ndarray, threshold: float = 0.9999) -> List[int]:
    """
    检测视频中的场景变化点
    
    参数:
        frames (np.ndarray): 视频帧数组
        threshold (float): 场景变化的阈值，取值[0, 1]，0为完全相同，1为完全不同
        
    返回:
        List[int]: 场景变化的帧索引列表
    """
    scene_changes = [0]  # 第一帧总是被认为是场景的开始
    prev_frame = frames[0]
    
    for i, frame in enumerate(frames[1:], start=1):
        diff = np.mean(np.abs(frame.astype(float) - prev_frame.astype(float)))
        if diff > threshold:
            scene_changes.append(i)
        prev_frame = frame
    
    return scene_changes

def split_video_by_scenes(
    frames: np.ndarray,
    fps: float,
    scene_changes: List[int],
    output_dir: str
) -> None:
    """
    根据场景变化点将视频拆分为多个片段并保存
    
    参数:
        frames (np.ndarray): 视频帧数组
        fps (float): 视频的帧率
        scene_changes (List[int]): 场景变化的帧索引列表
        output_dir (str): 输出文件夹的路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(len(scene_changes) - 1):
        start_frame = scene_changes[i]
        end_frame = scene_changes[i + 1]
        segment_frames = frames[start_frame:end_frame]
        
        output_path = output_dir / f"scene_{i+1}.mp4"
        save_video_mp4(segment_frames, output_path, fps)
        print(f"保存场景 {i+1} 到 {output_path}")

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
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError("无法创建输出视频文件")
    
    try:
        for frame in frames:
            out.write(frame)
    finally:
        out.release()

def extract_key_frames(frames: np.ndarray, scene_changes: List[int]) -> List[np.ndarray]:
    """
    提取关键帧
    
    参数:
        frames (np.ndarray): 视频帧数组
        scene_changes (List[int]): 场景变化的帧索引列表
        
    返回:
        List[np.ndarray]: 关键帧列表
    """
    key_frames = []
    for i in range(len(scene_changes) - 1):
        start_frame = scene_changes[i]
        end_frame = scene_changes[i + 1]
        key_frame = frames[start_frame]  # 选择每个场景的第一个帧作为关键帧
        key_frames.append(key_frame)
    
    return key_frames

def save_key_frames(key_frames: List[np.ndarray], output_dir: str) -> None:
    """
    保存关键帧为图像文件
    
    参数:
        key_frames (List[np.ndarray]): 关键帧列表
        output_dir (str): 输出文件夹的路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, key_frame in enumerate(key_frames):
        output_path = output_dir / f"key_frame_{i+1}.jpg"
        cv2.imwrite(str(output_path), key_frame)
        print(f"保存关键帧 {i+1} 到 {output_path}")

# 使用示例
if __name__ == "__main__":
    try:
        # 读取视频片段
        frames, fps = read_video_segment(
            video_path="./agan.mp4",
            start_time=2.0,
            end_time=90
        )
        print(f"成功读取视频片段，共 {len(frames)} 帧，FPS: {fps}")
        
        # 检测场景变化点
        scene_changes = detect_scene_changes(frames)
        print(f"检测到 {len(scene_changes)} 个场景变化点: {scene_changes}")
        
        # 新建输出文件夹
        output_dir = "output"
        
        # # 根据场景变化点拆分视频并保存
        # split_video_by_scenes(frames, fps, scene_changes, output_dir)
        # print("视频已成功拆分并保存")
        
        # 提取关键帧
        key_frames = extract_key_frames(frames, scene_changes)
        print(f"提取到 {len(key_frames)} 个关键帧")
        
        # 保存关键帧
        save_key_frames(key_frames, output_dir)
        print("关键帧已成功保存")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")