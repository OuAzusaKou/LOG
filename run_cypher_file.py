from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def execute_cypher_from_file(self, file_path):
        # 读取Cypher文件
        with open(file_path, 'r', encoding='utf-8') as file:
            cypher_query = file.read()
            
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return result.single()

# 使用示例
if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "your_password"

    try:
        conn = Neo4jConnection(uri, user, password)
        
        # 执行cypher文件
        result = conn.execute_cypher_from_file('create_relationships.cypher')
        print(f"关系创建完成，总数: {result['total']}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        
    finally:
        conn.close() 