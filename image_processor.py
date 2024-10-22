import os
import json
from datetime import datetime
from openai import OpenAI
import base64
import ssl

class ImageProcessor:
    def __init__(self, client):
        self.client = client

    def process_image(self, image_path):
        # 读取图片并转换为 base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # 提取时间戳
        timestamp = os.path.getmtime(image_path)
        date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

        # 使用 OpenAI API 分析图片
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"""分析这张图片,提供以下信息:\n1. 地点\n2. 人物数量\n3. 主要事件(最多3个)\n4. 图片描述(简短),使用JSON格式回复
                         示例如下：
                         {{
                            "date": "2024-02-20",
                            "location": "北京",
                            "people": 2,
                            "events": ["会议", "晚餐"],
                            "description": "这是一张在会议室拍摄的照片，显示了两个人在讨论问题。"
                         }}
                         """},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=3000,
            response_format={ "type": "json_object" }
        )

        # 解析 API 响应
        analysis = response.choices[0].message.content
        analysis_dict = json.loads(analysis)

        return {
            'date': date,
            'location': analysis_dict.get('location', None),
            'people': analysis_dict.get('people', 0),
            'events': analysis_dict.get('events', []),
            'description': analysis_dict.get('description', '')
        }



    def process_image_folder(self, folder_path):
        image_data = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                data = self.process_image(image_path)
                image_data.append(data)
        return image_data

# 使用示例
if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['HTTP_PROXY'] = 'http://192.168.224.76:7890'
    os.environ['HTTPS_PROXY'] = 'http://192.168.224.76:7890'

    client = OpenAI(
        api_key="sk-EjTppzE0xnr61QCj0d0MrZREohrwbV8xoMvOlvpw35g61vVG",
        base_url="https://api.openai-proxy.org/v1",
    )

    processor = ImageProcessor(client)
    folder_path = 'img_record_file'
    image_data = processor.process_image_folder(folder_path)

    # 将结果保存为 JSON 文件
    with open('image_data.json', 'w', encoding='utf-8') as f:
        json.dump(image_data, f, ensure_ascii=False, indent=4)

    print("图片处理完成,结果已保存到 image_data.json")
