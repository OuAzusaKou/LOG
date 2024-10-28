import json
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# from json2adjmatrix import json2adj
# from snn_graph_query_20240801.snn_active import active_node

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_npy_vector(npy_path):
    return np.load(npy_path)

def find_top_k_similar(embeddings, query_vector, top_k=5, model_type='bge'):
    similarities = []
    if model_type =='bgem3':
        for embedding in embeddings:
            vector = load_npy_vector(os.path.join('./title_embedding_bge',embedding+'.npy'))
            similarity = cosine_similarity([query_vector], [vector])[0][0]
            similarities.append((embedding, similarity))
    elif model_type == 'piccolo':
        for embedding in embeddings:
            vector = load_npy_vector(os.path.join('./title_embedding',embedding+'.npy'))
            similarity = cosine_similarity([query_vector], [vector])[0][0]
            similarities.append((embedding, similarity))
    elif model_type == 'bce':
        for embedding in embeddings:
            vector = load_npy_vector(os.path.join('./title_embedding_bce',embedding+'.npy'))
            similarity = cosine_similarity(query_vector, vector)[0][0]
            similarities.append((embedding, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def extract_embeddings(data):
    embeddings = []
    for item in data:
        if 'embeddings' in item:
            embeddings.extend(item['embeddings'])
        if 'subheaders' in item:
            embeddings.extend(extract_embeddings(item['subheaders']))
    return embeddings

def querypath2node(json_file_path, query_embeddings, top_k=5, model_type = 'bge'):
    data = load_json(json_file_path)
    query_vector = query_embeddings
    
    all_embeddings = extract_embeddings(data)
    
    top_k_similar = find_top_k_similar(all_embeddings, query_vector, top_k, model_type)
    
    # 保存结果
    top_k_embeddings = [embedding for embedding, similarity in top_k_similar]
    with open('top_k_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(top_k_embeddings, f, ensure_ascii=False, indent=4)
    
    print("Top-k embeddings have been saved to 'top_k_embeddings.json'")
    return top_k_embeddings


# def titlecode2context(json_file_path,md_dir,titlecode):

#     get_context_from_title(md_dir,titles)



#     return context


# def get_text_under_title(md_content, title):
#     # 创建匹配指定标题的正则表达式
#     title_pattern = re.compile(r'^(#{1,9})\s+' + re.escape(title) + r'\s*(.*?)$', re.MULTILINE)
#     # 创建匹配任何标题的正则表达式
#     any_title_pattern = re.compile(r'^(#{1,9})\s+.*$', re.MULTILINE)

#     # 查找指定标题
#     title_match = title_pattern.search(md_content)
#     if not title_match:
#         return None  # 如果没有找到标题，返回None

#     # 找到标题的级别和起始位置
#     title_level = len(title_match.group(1))
#     start_pos = title_match.end()

#     # 从标题之后开始，查找下一个同级或更高级标题
#     next_title_match = any_title_pattern.search(md_content, start_pos)
#     while next_title_match:
#         next_title_level = len(next_title_match.group(1))
#         if next_title_level <= title_level:
#             end_pos = next_title_match.start()
#             break
#         next_title_match = any_title_pattern.search(md_content, next_title_match.end())
#     else:
#         end_pos = len(md_content)  # 如果没有找到下一个同级或更高级标题，读取到文件末尾

#     # 提取标题下的文本内容
#     text_under_title = md_content[start_pos:end_pos].strip()
#     return text_under_title


# def get_context_from_title(directory,titles):
#     context = []
#     for title in titles:
#         file_name = title.split('/')[0]+'.md'
#         with open(os.path.join(directory,file_name), 'r', encoding='utf-8') as file:
#             md_content = file.read()
#         text = get_text_under_title(md_content, title.split('/')[-1])
#         context.append(text)
#     return context


def get_context_by_titles(data, titles):
    result = {}
    
    def search_subheaders(subheaders):
        for subheader in subheaders:
            if subheader['title'] in titles:
                result[subheader['title']] = subheader['context']
            search_subheaders(subheader['subheaders'])
    
    for item in data:
        if item['title'] in titles:
            result[item['title']] = item['context']
        search_subheaders(item['subheaders'])
    
    return result

# def extract_embeddings_and_titles(data, parent_title=None, result=None):
#     if result is None:
#         result = []

#     for item in data:
#         title = item['title']
#         embeddings = item.get('embeddings', [])
        
#         for embedding in embeddings:
#             result.append((embedding, title, parent_title))
        
#         subheaders = item.get('subheaders', [])
#         extract_embeddings_and_titles(subheaders, title, result)
    
#     return result

def index_by_number(dictionary):
    indexed_dict = {}
    for key, value in dictionary.items():
        number = key.split('/')[0]
        if number not in indexed_dict:
            indexed_dict[number] = []
        indexed_dict[number].append((key, value))
    return indexed_dict



def get_context_by_node(data, nodes):
    result = {}
    
    def search_subheaders(subheaders):
        for subheader in subheaders:
            if subheader['embeddings'] and subheader['embeddings'][0] in nodes:
                result[subheader['embeddings'][0]] = subheader['title']+':'+subheader['context']
            search_subheaders(subheader['subheaders'])
    
    for item in data:
        if item['embeddings'] and item['embeddings'][0] in nodes:
            result[item['embeddings'][0]] = item['title']+':'+ item['context']
        search_subheaders(item['subheaders'])
    
    return result

if __name__ == "__main__":
    json_file_path = 'toc.json'  # 替换为你的json文件路径
    # query_vector_path = 'title_embedding/0-2-1_c_0.npy'  # 替换为你的查询向量路径
    # top_k_embeddings = querypath2node(json_file_path, query_vector_path, top_k=5)
    # adj,title2index = json2adj(json_file_path)
    # node_title = top_k_embeddings[0]
    # if '_c' in node_title:
    #     node_title = node_title.split('_c')[0]
    # index_dict = index_by_number(title2index)
    # output = active_node(adj,10,index_dict[node_title][0][1])
    # node_index = output[:,2].cpu().numpy().tolist()
    # name_list = list(title2index.items())
    # title_name_list = []
    # for node in node_index:
    #     title_name_list.append(name_list[node][0].split('/')[1])
    # print(title_name_list)
    # data = load_json(json_file_path)
    # context = get_context_by_titles(data,title_name_list)
    # print(context)
    # # for title_code in title_name_list:
    # #     doc_code = int(title_code.strip('-')[0])
        
    # #     doc_name =
    # #     if len(title_code.strip('-'))>0:
    # #         title_code = int(title_code.strip('-')[-1])

    # # print(title_name_list)

    # context = get_context_by_node(data, top_k_embeddings)
    # print(context)

