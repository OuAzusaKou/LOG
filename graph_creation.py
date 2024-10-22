import os
import json
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def load_npy_vector(npy_path):
    return np.load(npy_path)

def create_graph_from_json(json_file_path, graph_image_path, graph_json_path, adjacency_matrix_path):
    # 加载JSON数据
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建图
    G = nx.Graph()

    # 创建时刻节点并连接相邻时刻
    dates = sorted(data.keys())
    for i, date in enumerate(dates):
        if date != "overall_relationships":
            G.add_node(date, type='时刻', info=data[date])
            if i > 0:
                prev_date = dates[i-1]
                if prev_date != "overall_relationships":
                    G.add_edge(prev_date, date)
                    G.add_edge(date, prev_date)  # 添加双向连接

    # 创建人物、地点和事件节点,并连接到相应的时刻节点
    for date, events in data.items():
        if date != "overall_relationships":
            for event in events:
                # 添加人物节点
                person = event['person']
                if not G.has_node(person):
                    embedding_path = os.path.join('./chat_analysis_embedding_bge', person + '.npy')
                    G.add_node(person, type='人物', info=embedding_path)
                G.add_edge(person, date)
                G.add_edge(date, person)  # 添加双向连接
                
                # 添加地点节点
                if event['location']:
                    location = event['location']
                    if not G.has_node(location):
                        embedding_path = os.path.join('./chat_analysis_embedding_bge', 'location_' + location + '.npy')
                        G.add_node(location, type='地点', info=embedding_path)
                    G.add_edge(location, date)
                    G.add_edge(date, location)  # 添加双向连接
                
                # 添加事件节点
                for e in event['events']:
                    if not G.has_node(e):
                        embedding_path = os.path.join('./chat_analysis_embedding_bge', 'event_' + e + '.npy')
                        G.add_node(e, type='事件', info=embedding_path)
                    G.add_edge(e, date)
                    G.add_edge(date, e)  # 添加双向连接
                
                # 连接人物节点
                for rel, info in event['relationships'].items():
                    if not G.has_node(rel):
                        embedding_path = os.path.join('./chat_analysis_embedding_bge', rel + '.npy')
                        G.add_node(rel, type='人物', info=embedding_path)
                    G.add_edge(person, rel, relation=info['关系'])
                    G.add_edge(rel, person, relation=info['关系'])  # 添加双向连接

    # 合并相似的“地点”型节点
    merge_similar_nodes(G, '地点', data)

    # 合并相似的“事件”型节点
    merge_similar_nodes(G, '事件', data)

    # 打印图的基本信息
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边的数量: {G.number_of_edges()}")

    # 将图片保存为png格式
    nx.draw(G, with_labels=True)
    plt.savefig(graph_image_path)

    # 将图结构保存为JSON格式
    graph_data = nx.node_link_data(G)
    with open(graph_json_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    # 创建并保存邻接矩阵
    adj_matrix = nx.adjacency_matrix(G).todense()
    np.savetxt(adjacency_matrix_path, adj_matrix, delimiter=',', fmt='%d')

    print(f"图结构已保存为 {graph_json_path}")
    print(f"邻接矩阵已保存为 {adjacency_matrix_path}")

def merge_similar_nodes(G, node_type, data, similarity_threshold=0.8):
    # 获取所有指定类型的节点
    nodes = [node for node, attr in G.nodes(data=True) if attr['type'] == node_type]
    
    # 遍历节点并合并相似节点
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            # 加载节点的嵌入向量
            if node_type == '地点':
                node1_address = os.path.join('./chat_analysis_embedding_bge', 'location_' + node1 + '.npy')
                node2_address = os.path.join('./chat_analysis_embedding_bge', 'location_' + node2 + '.npy')
            elif node_type == '事件':
                node1_address = os.path.join('./chat_analysis_embedding_bge', 'event_' + node1 + '.npy')
                node2_address = os.path.join('./chat_analysis_embedding_bge', 'event_' + node2 + '.npy')
            else:
                node1_address = os.path.join('./chat_analysis_embedding_bge', node1 + '.npy')
                node2_address = os.path.join('./chat_analysis_embedding_bge', node2 + '.npy')
            node1_vector = load_npy_vector(node1_address)
            node2_vector = load_npy_vector(node2_address)
            
            # 计算相似度
            similarity = check_similarity(node1_vector, node2_vector)
            if similarity > similarity_threshold:
                # 合并节点
                print(f"合并节点 {node1} 和 {node2}")
                merge_nodes(G, node1, node2)

def load_node_embedding(node_address):
    # 假设节点名称与嵌入文件名一致  
    # embedding_path = os.path.join('./chat_analysis_embedding_bge', node + '.npy')
    return load_npy_vector(node_address)

def check_similarity(node1_vector, node2_vector):
    # 计算两个节点向量的余弦相似度
    similarity = cosine_similarity([node1_vector], [node2_vector])[0][0]
    return similarity

def merge_nodes(G, node1, node2):
    # 将node2的所有边转移到node1
    for neighbor in list(G.neighbors(node2)):
        G.add_edge(node1, neighbor)
    # 移除node2
    G.remove_node(node2)

# 示例调用
# create_graph_from_json('chat_analysis_emb.json', 'chat_graph.png', 'graph_structure.json', 'adjacency_matrix.csv')
if __name__ == "__main__":
    create_graph_from_json('chat_analysis_emb.json', 'chat_graph.png', 'graph_structure.json', 'adjacency_matrix.csv')
