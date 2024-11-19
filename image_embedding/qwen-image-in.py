import os
import dashscope

def qwen_image(image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": "描述图像"}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model='qwen-vl-max',
        messages=messages
        )
    
    print(response["output"]["choices"][0]["message"]["content"][0]["text"])

if __name__ == '__main__':
    # 指定本地图片路径
    image_path = '/home/xiaodong.hu/tools/example.png'
    qwen_image(image_path)