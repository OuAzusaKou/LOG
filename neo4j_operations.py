from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def add_node_with_relationship(self, new_node_id, new_node_name, existing_node_id):
        with self.driver.session() as session:
            # 创建新节点并建立关系的Cypher查询
            query = """
            CREATE (n:Node {id: $new_id, name: $new_name})
            WITH n
            MATCH (existing:Node {id: $existing_id})
            MERGE (n)-[:恋人]->(existing)
            RETURN n, existing
            """
            
            result = session.run(query, 
                               new_id=new_node_id,
                               new_name=new_node_name,
                               existing_id=existing_node_id)
            return result.single()

# 使用示例
if __name__ == "__main__":
    # 连接参数
    uri = "bolt://localhost:7687"  # Neo4j默认地址和端口
    user = "neo4j"                 # 默认用户名
    password = "your_password"     # 你的密码

    try:
        # 创建连接
        conn = Neo4jConnection(uri, user, password)
        
        # 添加新节点并创建关系
        new_node_id = "new_person_001"
        new_node_name = "张三"
        existing_node_id = "existing_person_001"
        
        result = conn.add_node_with_relationship(
            new_node_id,
            new_node_name,
            existing_node_id
        )
        
        print("节点创建成功！")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        
    finally:
        # 关闭连接
        conn.close() 