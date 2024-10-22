import json
import os
import numpy as np
from title2embedding import get_sentence_embedding
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from BCEmbedding import EmbeddingModel

def process_json_to_embeddings(json_path, model_type, output_json_path = 'toc_emb.json'):
    device = 'cuda:1'

    # 配置使用的模型
    if model_type == 'bgem3':
        encoder_model = BGEM3FlagModel('./bge-m3', use_fp16=True)
        embedding_folder = './title_embedding_bge'
    elif model_type == 'piccolo':
        encoder_model = SentenceTransformer('./snn_rag/piccolo')
        embedding_folder = './title_embedding_piccolo'
    elif model_type == 'bce':
        encoder_model = EmbeddingModel(model_name_or_path="./bce-base")
        embedding_folder = './title_embedding_bce'
    else:
        raise ValueError("不支持的模型类型")

    # 确保embedding文件夹存在
    os.makedirs(embedding_folder, exist_ok=True)

    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    def split_string_by_length(s, length=512):
        return [s[i:i+length] for i in range(0, len(s), length)]

    # 函数：递归遍历 JSON 数据并添加 `embeddings` 字段
    def add_embeddings(node, address=''):
        if isinstance(node, dict):
            node['embeddings'] = []
            node['embeddings'].append(address)  # 添加 `embeddings` 字段
            sentence = node['title']
            
            if model_type == 'bgem3':
                embeddings = encoder_model.encode(sentence)['dense_vecs']
            elif model_type in ['piccolo', 'bce']:
                embeddings = get_sentence_embedding(sentence, tokenizer=None, model=None, encoder_model=encoder_model)
                embeddings = embeddings.cpu().numpy()

            np.save(os.path.join(embedding_folder, address), embeddings)
            
            if ('context' in node) and len(node['context']) > 0:
                sentence = node['context']
                sentence_list = split_string_by_length(sentence)
                for i, s in enumerate(sentence_list):
                    node['embeddings'].append(f"{address}_c_{i}")
                    
                    if model_type == 'bgem3':
                        embeddings = encoder_model.encode(s)['dense_vecs']
                    elif model_type in ['piccolo', 'bce']:
                        embeddings = get_sentence_embedding(s, tokenizer=None, model=None, encoder_model=encoder_model)
                        embeddings = embeddings.cpu().numpy()
                    
                    np.save(os.path.join(embedding_folder, f"{address}_c_{i}"), embeddings)

            for key, value in node.items():
                if key == 'subheaders' and isinstance(value, list):
                    for idx, subheader in enumerate(value):
                        new_address = f"{address}-{idx}" if address else str(idx)
                        add_embeddings(subheader, new_address)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                new_address = f"{address}-{idx}" if address else str(idx)
                add_embeddings(item, new_address)

    # 添加 `embeddings` 字段到每个标题
    add_embeddings(data)

    # 将修改后的 JSON 数据写回文件
    
    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"完成！修改后的 JSON 数据已保存到 {output_json_path}，embedding 已保存到 {embedding_folder} 文件夹")

# 使用示例
if __name__ == "__main__":
    json_path = 'toc.json'
    model_type = 'bce'  # 可以是 'bgem3' 或 'piccolo' 'bce'
    process_json_to_embeddings(json_path, model_type)