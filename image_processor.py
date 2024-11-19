import os
import json
from datetime import datetime
import numpy as np
from openai import OpenAI
import base64
import ssl
from FlagEmbedding import BGEM3FlagModel
from integrate_image_data import get_embedding



def add_embeddings(node, date, person_index):
            if isinstance(node, dict):
                node['embeddings'] = []

                # 处理日期的embedding
                if 'date' in node:
                    date_address = f"date_{node['date']}_{person_index}"
                    node['embeddings'].append(date_address)
                    date_embeddings = get_embedding(node['date'])
                    np.save(os.path.join(embedding_folder, date_address), date_embeddings)

                # 处理人物的embedding
                if 'person' in node:
                    person_address = f"person_{node['person']}"
                    node['embeddings'].append(person_address)
                    person_embeddings = get_embedding(node['person'])
                    np.save(os.path.join(embedding_folder, person_address), person_embeddings)

                # 处理地点的embedding
                if 'location' in node:
                    location_address = f"location_{node['location']}"
                    node['embeddings'].append(location_address)
                    location_embeddings = get_embedding(node['location'])
                    np.save(os.path.join(embedding_folder, location_address), location_embeddings)
                    
                if 'events' in node:
                    for event in node['events']:
                        event_address = f"event_{event}"
                        node['embeddings'].append(event_address)
                        event_embeddings = get_embedding(event)
                        np.save(os.path.join(embedding_folder, event_address), event_embeddings)
                # 处理关系的embedding
                if 'relationships' in node:
                    for rel_person, rel_info in node['relationships'].items():
                        rel_address = f"relation_{rel_info['关系']}"
                        node['embeddings'].append(rel_address)
                        rel_embeddings = get_embedding(rel_info['关系'])
                        np.save(os.path.join(embedding_folder, rel_address), rel_embeddings)

class ImageProcessor:
    def __init__(self, client,encoder_model):
        self.client = client
        self.encoder_model = encoder_model

    def process_image(self, image_path, embedding_folder):
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

        date_address = f"date_{date}"
        date_embeddings = get_embedding(date,encoder_model=self.encoder_model)
        np.save(os.path.join(embedding_folder, date_address), date_embeddings)

        location_address = f"location_{analysis_dict.get('location', None)}"
        location_embeddings = get_embedding(analysis_dict.get('location', None),encoder_model=self.encoder_model)
        np.save(os.path.join(embedding_folder, location_address), location_embeddings)

        events = analysis_dict.get('events', [])
        for event in events:
            event_address = f"event_{event}"
            event_embeddings = get_embedding(event,encoder_model=self.encoder_model)
            np.save(os.path.join(embedding_folder, event_address), event_embeddings)
        

        return {
            'date': date,
            'location': analysis_dict.get('location', None),
            'people': analysis_dict.get('people', 0),
            'events': analysis_dict.get('events', []),
            'description': analysis_dict.get('description', ''),
            'image_path': image_path,
            'embeddings': [date_address, location_address] + [f"event_{event}" for event in events]
        }



    def process_image_folder(self, folder_path,embedding_folder):
        image_data = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                data = self.process_image(image_path,embedding_folder)
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
    encoder_model = BGEM3FlagModel('./bge-m3', use_fp16=True)
    processor = ImageProcessor(client,encoder_model=encoder_model)
    folder_path = 'img_record_file'
    embedding_folder = 'story_analysis_embedding_bge'
    image_data = processor.process_image_folder(folder_path,embedding_folder)

    # 将结果保存为 JSON 文件
    with open('image_data.json', 'w', encoding='utf-8') as f:
        json.dump(image_data, f, ensure_ascii=False, indent=4)

    print("图片处理完成,结果已保存到 image_data.json")
