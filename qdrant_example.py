from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

# 连接到 Qdrant
client = QdrantClient("localhost", port=6333)

# 创建一个集合（collection）
collection_name = "my_collection"
collections = client.get_collections().collections
existing_collections = [collection.name for collection in collections]

if collection_name not in existing_collections:
    print(f"创建新集合: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=4,  # 向量维度
            distance=models.Distance.COSINE  # 距离度量方式
        )
    )
else:
    print(f"集合 {collection_name} 已存在，跳过创建步骤")

# 准备一些示例数据
vectors = [
    np.random.rand(4).tolist(),  # 第一条记录的向量
    np.random.rand(4).tolist(),  # 第二条记录的向量
    np.random.rand(4).tolist(),  # 第三条记录的向量
]

# 准备对应的元数据
payload = [
    {"name": "记录1", "category": "类别A"},
    {"name": "记录2", "category": "类别B"},
    {"name": "记录3", "category": "类别A"},
]

# 插入数据（不指定 ID）
client.upsert(
    collection_name=collection_name,
    points=models.Batch(
        ids=[1,2,3],
        vectors=vectors,
        payloads=payload
    )
)

# 搜索最相似的2条记录
search_vector = np.random.rand(4).tolist()
search_result = client.search(
    collection_name=collection_name,
    query_vector=search_vector,
    limit=2
)

# 打印搜索结果
print("搜索结果：")
for result in search_result:
    print(f"ID: {result.id}")
    print(f"相似度分数: {result.score}")
    print(f"元数据: {result.payload}")
    print("---")

# 删除单个记录
print("\n删除ID为1的记录")
client.delete(
    collection_name=collection_name,
    points_selector=models.PointIdsList(
        points=[1]
    )
)

# 验证删除结果
remaining_points = client.scroll(
    collection_name=collection_name,
    limit=10
)[0]
print(f"剩余记录数量: {len(remaining_points)}")

# 清空整个集合
print("\n清空整个集合")
client.delete(
    collection_name=collection_name,
    points_selector=models.FilterSelector(
        filter=models.Filter()  # 空过滤器表示匹配所有记录
    )
)

# 验证集合是否已清空
empty_check = client.scroll(
    collection_name=collection_name,
    limit=10
)[0]
print(f"清空后的记录数量: {len(empty_check)}")

# 按条件过滤搜索
filtered_results = client.search(
    collection_name=collection_name,
    query_vector=search_vector,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="category",
                match=models.MatchValue(value="类别A")
            )
        ]
    ),
    limit=2
)

print("\n按类别A过滤的搜索结果：")
for result in filtered_results:
    print(f"ID: {result.id}")
    print(f"相似度分数: {result.score}")
    print(f"元数据: {result.payload}")
    print("---")

# 查看生成的 ID
results = client.scroll(
    collection_name=collection_name,
    limit=10
)[0]
for point in results:
    print(f"自动生成的ID: {point.id}") 