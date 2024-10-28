

import json
import os

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from node_filter import bfs_k_steps_with_paths
from query2ac_node import load_json
from query_analysis import QueryAnalysisSystem, setup_models
# from rag_system import setup_models
from story_rag_system import find_top_k_similar


def find_top_k_similar_in_folder(query_vector, folder_path, top_k=5, data_type='entity'):
    similarities = []
    for file_name in os.listdir(folder_path):
        if data_type == 'entity':
            if file_name.startswith(('location', 'person', 'date', 'event')) and file_name.endswith('.npy'):
                vector = np.load(os.path.join(folder_path, file_name))
                similarity = cosine_similarity([query_vector], [vector])[0][0]
                # 去掉前缀和后缀
                clean_name = file_name.split('_', 1)[-1].rsplit('.', 1)[0]
                similarities.append((clean_name, similarity))
        elif data_type == 'relationship':
            if file_name.startswith(('relation')) and file_name.endswith('.npy'):
                vector = np.load(os.path.join(folder_path, file_name))
                similarity = cosine_similarity([query_vector], [vector])[0][0]
                # 去掉前缀和后缀
                clean_name = file_name.split('_', 1)[-1].rsplit('.', 1)[0]
                similarities.append((clean_name, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def get_top_3_matches(query, json_data, encoder_model,query_analysis_system):

    # 获取查询的嵌入
    
    analysis_result = query_analysis_system.extract_entities_and_relationships(query).choices[0].message.content
    analysis_result = json.loads(analysis_result)
    entity_embeddings, relationship_embeddings = query_analysis_system.get_embeddings(analysis_result)
    
    # 收集所有嵌入
    all_embeddings = []
    for date_entries in json_data['daily_records'].values():
        for person_entries in date_entries:
            for person, entries in person_entries.items():
                if person != '原文':        
                    for entry in entries:
                        all_embeddings.extend(entry['embeddings'])
    
    # 计算每个查询嵌入的前3个匹配
    top_3_entity_matches = []
    top_3_relationship_matches = []
    for query_embedding in entity_embeddings.values():
        top_3_similar = find_top_k_similar_in_folder(query_embedding, 'story_analysis_embedding_bge', 3, data_type='entity')
        top_3_entity_matches.append(top_3_similar)
    for query_embedding in relationship_embeddings.values():
        top_3_similar = find_top_k_similar_in_folder(query_embedding, 'story_analysis_embedding_bge', 3, data_type='relationship')
        top_3_relationship_matches.append(top_3_similar)
    
    # 将所有的关键字组成集合并去重
    top_3_entity_matches_flat = [embedding for matches in top_3_entity_matches for embedding, _ in matches]
    top_3_entity_matches_set = set(top_3_entity_matches_flat)

    # 对top_3_relationship_matches进行去重
    top_3_relationship_matches_flat = [embedding for matches in top_3_relationship_matches for embedding, _ in matches]
    top_3_relationship_matches_set = set(top_3_relationship_matches_flat)


    with open('graph_structure.json', 'r', encoding='utf-8') as f:
        graph_structure = json.load(f)

    # Example usage
    start_nodes = top_3_entity_matches_set  # Replace with your starting node IDs
    k = 1  # Number of steps

    visited_nodes,paths = bfs_k_steps_with_paths(graph_structure, start_nodes, k)

    # # 获取相应的文本
    # matched_texts = []
    # for matches in top_3_entity_matches_set:
    #     for embedding, _ in matches:
    #         for date, entries in json_data['daily_records'].items():
    #             for person_entries in entries:
    #                 for person, person_entries in person_entries.items():
    #                     if person != '原文':
    #                         for entry in person_entries:
    #                             if embedding in entry['embeddings']:
    #                                 context = f"日期: {entry['date']}\n"
    #                                 context += f"人物: {entry['person']}\n"
    #                                 context += f"地点: {entry['location']}\n"
    #                                 context += f"事件: {', '.join(entry['events'])}\n"
    #                                 for rel, details in entry['relationships'].items():
    #                                     context += f"与{rel}的关系: {details['关系']}, 互动: {details['互动']}\n"
    #                                 matched_texts.append(context)
    #                                 break
    
    return visited_nodes,paths


if __name__ == '__main__':

    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['HTTP_PROXY'] = 'http://192.168.224.76:7890'
    os.environ['HTTPS_PROXY'] = 'http://192.168.224.76:7890'

    client = OpenAI(
        api_key="sk-EjTppzE0xnr61QCj0d0MrZREohrwbV8xoMvOlvpw35g61vVG", # 如果您没有配置环境变量，请��此处用您的API Key进行替换
        base_url="https://api.openai-proxy.org/v1",  # 填写DashScope服务的base_url
    )

    encoder_model = setup_models()

    query_analysis_system = QueryAnalysisSystem(chat_folder='human_chart', client=client, encoder_model=encoder_model)
    
    json_data = load_json('chat_analysis_emb.json')
    while True:
        query = input('请输入问题：')
        print(get_top_3_matches(query, json_data, encoder_model,query_analysis_system))

    
