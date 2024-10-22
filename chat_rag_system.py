import os
import json
import ssl
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from datetime import datetime
from collections import deque

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_npy_vector(npy_path):
    return np.load(npy_path)

def setup_models():
    device = 'cuda:0'
    encoder_model = BGEM3FlagModel('./bge-m3', use_fp16=True)
    return encoder_model

def get_response(message, contexts, history_messages):
    client = OpenAI(
        api_key="sk-EjTppzE0xnr61QCj0d0MrZREohrwbV8xoMvOlvpw35g61vVG",
        base_url="https://api.openai-proxy.org/v1",
    )
    messages = [
        {'role': 'system', 'content': '你是一个名叫小Q的具有幽默感的聊天陪伴角色。你应该用轻松幽默的方式回应用户的消息，并保持对话的趣味性。始终用中文回复。其中context中是用户之前的相关信息。'},
    ]
    messages.extend(history_messages)
    messages.append({'role': 'user', 'content': str({'context': contexts, 'message': message})})
    
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return completion.choices[0].message.content

def find_top_k_similar(embeddings, query_vector, top_k=5):
    similarities = []
    for embedding in embeddings:
        vector = load_npy_vector(os.path.join('./chat_analysis_embedding_bge', embedding + '.npy'))
        similarity = cosine_similarity([query_vector], [vector])[0][0]
        similarities.append((embedding, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def get_context(question, json_data, encoder_model, top_k=5):
    query_embeddings = encoder_model.encode(question)['dense_vecs']
    
    all_embeddings = []
    for date_entries in json_data.values():
        if isinstance(date_entries, list):  # 处理日期条目
            for entry in date_entries:
                all_embeddings.extend(entry['embeddings'])
        elif isinstance(date_entries, dict) and 'embeddings' in date_entries:  # 处理overall_relationships
            all_embeddings.extend(date_entries['embeddings'])
    
    top_k_similar = find_top_k_similar(all_embeddings, query_embeddings, top_k)
    
    contexts = []
    for embedding, _ in top_k_similar:
        if embedding.startswith('overall_'):
            # 处理overall_relationships
            overall_context = "整体关系概览:\n"
            for person, details in json_data['overall_relationships'].items():
                if person != 'embeddings':
                    overall_context += f"{person}: 关系 - {details['关系']}, 互动 - {details['互动']}"
                    if '最近互动' in details:
                        overall_context += f", 最近互动 - {details['最近互动']}"
                    overall_context += "\n"
            contexts.append(overall_context)
        else:
            # 处理日期条目
            for date, entries in json_data.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if embedding in entry['embeddings']:
                            context = f"日期: {date}\n"
                            context += f"人物: {entry['person']}\n"
                            context += f"地点: {entry['location']}\n"
                            context += f"事件: {', '.join(entry['events'])}\n"
                            for rel, details in entry['relationships'].items():
                                context += f"与{rel}的关系: {details['关系']}, 互动: {details['互动']}\n"
                            contexts.append(context)
    
    return contexts

def save_chat_record(message, response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    chat_record = f"""时间戳：{timestamp}

你：{message}

时间戳：{timestamp}

小Q：{response}

"""
    with open('robot.txt', 'a', encoding='utf-8') as f:
        f.write(chat_record)

def get_chat_history(history, max_history=10):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": msg} 
            for i, msg in enumerate(history)]

def main():
    json_file_path = 'chat_analysis_emb.json'
    json_data = load_json(json_file_path)
    encoder_model = setup_models()

    print("欢迎来到小Q的幽默聊天室！输入'退出'结束对话。")

    chat_history = deque(maxlen=20)  # 存储最近20条消息(10轮对话)

    while True:
        message = input("你：")
        if message.lower() == '退出':
            break

        contexts = get_context(message, json_data, encoder_model)
        history_messages = get_chat_history(chat_history)
        response = get_response(message, contexts, history_messages)

        print(f'小Q：{response}\n')
        save_chat_record(message, response)
        
        # 更新聊天历史
        chat_history.append(message)
        chat_history.append(response)

    print("谢谢聊天，再见！")

if __name__ == "__main__":
    
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['HTTP_PROXY'] = 'http://192.168.224.76:7890'
    os.environ['HTTPS_PROXY'] = 'http://192.168.224.76:7890'

    main()
