import json
import os
import numpy as np
from FlagEmbedding import BGEM3FlagModel

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_embedding(text, encoder_model):
    return encoder_model.encode(text)['dense_vecs']

def integrate_image_data(chat_data, image_data, encoder_model, embedding_folder):
    for index, image in enumerate(image_data):
        date = image['date']
        if date not in chat_data:
            chat_data[date] = []
        
        # 生成唯一的地址，加入索引以区分同一天的不同图片
        address = f"{date}_image_{index}"
        
        entry = {
            "person": "本人",
            "date": date,
            "location": image['location'],
            "events": image['events'],
            "relationships": {},
            "embeddings": [address],
            "image_description": image['description'],
            "people_count": image['people']
        }
        
        # 生成并保存embedding
        info = f"日期: {date}, 地点: {image['location']}, 事件: {', '.join(image['events'])}, 描述: {image['description']}, 人数: {image['people']}"
        embeddings = get_embedding(info, encoder_model)
        np.save(os.path.join(embedding_folder, address), embeddings)
        
        chat_data[date].append(entry)
    
    return chat_data

if __name__ == '__main__':
    # 加载数据
    chat_data = load_json('chat_analysis_emb.json')
    image_data = load_json('image_data.json')

    # 设置模型和embedding文件夹
    device = 'cuda:1'
    encoder_model = BGEM3FlagModel('./bge-m3', use_fp16=True)
    embedding_folder = './chat_analysis_embedding_bge'

    # 确保embedding文件夹存在
    os.makedirs(embedding_folder, exist_ok=True)

    # 整合数据
    updated_chat_data = integrate_image_data(chat_data, image_data, encoder_model, embedding_folder)

    # 保存更新后的数据
    save_json(updated_chat_data, 'updated_chat_analysis_emb.json')

    print(f"完成！修改后的JSON数据已保存到updated_chat_analysis_emb.json，图片描述的embedding已保存到{embedding_folder}文件夹")
