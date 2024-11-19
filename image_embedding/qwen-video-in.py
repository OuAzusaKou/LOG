from http import HTTPStatus
import os
# dashscope版本需要不低于1.20.10
import dashscope


def get_image_paths(video_path):
    # 支持的图像文件扩展名
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    # 存储图像路径的列表
    image_paths = []
    
    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(video_path):
        for file in files:
            # 获取文件扩展名并转换为小写
            ext = os.path.splitext(file)[1].lower()
            
            # 检查文件是否是支持的图像格式
            if ext in valid_extensions:
                # 构建完整路径并添加到列表中
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
                
    return image_paths


def qwen_video(video_path):
    image_paths = get_image_paths(video_path)

    messages = [{"role": "user",
                 "content": [
                     {"video": image_paths},
                     {"text": "描述这个视频的具体过程"}]}]
                # 不支持直接输入视频格式文件，video使用连续多帧图像输入
    response = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model='qwen-vl-max-latest',
        messages=messages
    )

    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print(response.code)
        print(response.message)


if __name__ == '__main__':
    # 指定本地图片路径
    video_path = r'/home/xiaodong.hu/image_embedding/iceskater1'
    qwen_video(video_path)
