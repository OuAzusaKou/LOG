import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from BCEmbedding import EmbeddingModel

def process_chat_analysis_to_embeddings(json_path, model_type, output_json_path='chat_analysis_emb.json'):
    device = 'cuda:1'

    # 配置使用的模型
    if model_type == 'bgem3':
        encoder_model = BGEM3FlagModel('./bge-m3', use_fp16=True)
        embedding_folder = './chat_analysis_embedding_bge'
    elif model_type == 'piccolo':
        encoder_model = SentenceTransformer('./snn_rag/piccolo')
        embedding_folder = './chat_analysis_embedding_piccolo'
    elif model_type == 'bce':
        encoder_model = EmbeddingModel(model_name_or_path="./bce-base")
        embedding_folder = './chat_analysis_embedding_bce'
    else:
        raise ValueError("不支持的模型类型")

    # 确保embedding文件夹存在
    os.makedirs(embedding_folder, exist_ok=True)

    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    def get_embedding(text):
        if model_type == 'bgem3':
            return encoder_model.encode(text)['dense_vecs']
        elif model_type in ['piccolo', 'bce']:
            return encoder_model.encode(text).cpu().numpy()

    # 函数：递归遍历 JSON 数据并添加 `embeddings` 字段
    def add_embeddings(node, date, person_index):
        if isinstance(node, dict):
            node['embeddings'] = []
            
            # 生成唯一的地址
            address = f"{date}_{person_index}"
            node['embeddings'].append(address)

            # 将所有相关信息合并成一个字符串
            info = f"日期: {node.get('date', '')}, 人物: {node.get('person', '')}, 地点: {node.get('location', '')}, 事件: {', '.join(node.get('events', []))}"
            
            # 生成并保存embedding
            embeddings = get_embedding(info)
            np.save(os.path.join(embedding_folder, address), embeddings)

            # 处理地点的embedding
            location = node.get('location', '')
            if location:
                location_address = f"location_{location}"
                node['embeddings'].append(location_address)
                location_embeddings = get_embedding(location)
                np.save(os.path.join(embedding_folder, location_address), location_embeddings)

            # 处理事件的embedding
            events = node.get('events', [])
            for i, event in enumerate(events):
                event_address = f"event_{event}"
                node['embeddings'].append(event_address)
                event_embeddings = get_embedding(event)
                np.save(os.path.join(embedding_folder, event_address), event_embeddings)

            # 处理relationships
            if 'relationships' in node:
                for rel_person, rel_info in node['relationships'].items():
                    rel_address = f"{address}_rel_{rel_person}"
                    node['embeddings'].append(rel_address)
                    rel_info_str = f"关系: {rel_info.get('关系', '')}, 互动: {rel_info.get('互动', '')}"
                    rel_embeddings = get_embedding(rel_info_str)
                    np.save(os.path.join(embedding_folder, rel_address), rel_embeddings)

    # 遍历数据并添加embeddings
    for date, people_list in data.items():
        if date != 'overall_relationships':
            for index, person_data in enumerate(people_list):
                add_embeddings(person_data, date, index)
    
    # 处理overall_relationships
    if 'overall_relationships' in data:
        data['overall_relationships']['embeddings'] = []
        for person, rel_info in data['overall_relationships'].items():
            if person != 'embeddings':  # 跳过embeddings键
                address = f"overall_{person}"
                data['overall_relationships']['embeddings'].append(address)
                info = f"人物: {person}, 关系: {rel_info.get('关系', '')}, 互动: {rel_info.get('互动', '')}, 最近互动: {rel_info.get('最近互动', '')}"
                embeddings = get_embedding(info)
                np.save(os.path.join(embedding_folder, address), embeddings)

    # 将修改后的 JSON 数据写回文件
    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"完成！修改后的 JSON 数据已保存到 {output_json_path}，embedding 已保存到 {embedding_folder} 文件夹")

# 使用示例
if __name__ == "__main__":
    json_path = 'sorted_chat_analysis_results.json'
    model_type = 'bgem3'  # 可以是 'bgem3' 或 'piccolo' 或 'bce'
    process_chat_analysis_to_embeddings(json_path, model_type)
