import json
import csv

# 读取JSON文件
with open('graph_structure.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 保存节点到CSV
with open('nodes.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'type', 'info'])
    writer.writeheader()
    for node in data['nodes']:
        writer.writerow(node)

# 保存关系到CSV
with open('relationships.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['source', 'target', 'relation'])
    writer.writeheader()
    for link in data['links']:
        writer.writerow({
            'source': link['source'],
            'target': link['target'],
            'relation': link.get('relation')
        }) 