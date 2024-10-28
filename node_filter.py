import json
from collections import deque

def bfs_k_steps_with_paths(graph, start_nodes, k):
    visited = set(start_nodes)
    queue = deque([(node, 0, []) for node in start_nodes])  # (node, current_step, path)

    paths = []  # 用于存储所有路径

    while queue:
        current_node, current_step, current_path = queue.popleft()
        
        if current_step < k:
            for link in graph['links']:
                if link['source'] == current_node and link['target'] not in visited:
                    visited.add(link['target'])
                    if 'relation' in link:
                        new_path = current_path + [(link['source'], link['relation'], link['target'])]
                    else:
                        new_path = current_path + [(link['source'], link['target'])]
                    queue.append((link['target'], current_step + 1, new_path))
                    paths.append(new_path)
                elif link['target'] == current_node and link['source'] not in visited:
                    visited.add(link['source'])
                    if 'relation' in link:
                        new_path = current_path + [(link['target'], link['relation'], link['source'])]
                    else:
                        new_path = current_path + [(link['target'], link['source'])]
                    queue.append((link['source'], current_step + 1, new_path))
                    paths.append(new_path)

    return visited, paths

# Load the graph structure from JSON
if __name__ == "__main__":
    with open('graph_structure.json', 'r', encoding='utf-8') as f:
        graph_structure = json.load(f)

    # Example usage
    start_nodes = ["1946-03-01", "保罗·乔布斯"]  # Replace with your starting node IDs
    k = 2  # Number of steps
    visited_nodes, paths = bfs_k_steps_with_paths(graph_structure, start_nodes, k)
    print("访问过的节点:", visited_nodes)
    print("传播路径:")
    for path in paths:
        print(" -> ".join(f"{src} -[{rel}]-> {tgt}" for src, rel, tgt in path))
