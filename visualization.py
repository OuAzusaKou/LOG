import json
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import font_manager

# 设置中文字体
font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'  # 请替换为您系统中的中文字体路径
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# 读取JSON文件
with open('chat_analysis_results.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 创建图
G = nx.Graph()

# 添加节点和边
for date, records in data['daily_records'].items():
    for record in records:
        for person, info in record.items():
            # 将'你'和'本人'统一为'本人'
            if person in ['你', '本人']:
                person = '本人'
            G.add_node(person)
            for relationship_info in info:
                for related_person, details in relationship_info['relationships'].items():
                    # 将'你'和'本人'统一为'本人'
                    if related_person in ['你', '本人']:
                        related_person = '本人'
                    G.add_edge(person, related_person, relationship=details['关系'])

# 创建时间轴
dates = sorted(data['daily_records'].keys())
timeline = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 绘制图
plt.figure(figsize=(15, 10))

# 绘制关系图
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold',font_family=font_prop.get_name())

# 添加边标签
edge_labels = nx.get_edge_attributes(G, 'relationship')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_family=font_prop.get_name())

# 添加时间轴
ax = plt.gca()
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(len(timeline)))
ax2.set_yticklabels([date.strftime('%Y-%m-%d') for date in timeline])

plt.title('人物关系图与时间轴')
plt.tight_layout()
plt.savefig('relationship_graph.png')