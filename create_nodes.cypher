// 删除现有的数据
MATCH (n) DETACH DELETE n;

// 创建索引
CREATE INDEX node_id FOR (n:Node) ON (n.id);

// 创建节点
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row WHERE row.id IS NOT NULL
MERGE (n:Node {id: row.id})
SET n.type = row.type,
    n.info = row.info;

// 为不同类型的节点添加标签
MATCH (n:Node)
WHERE n.type = '时刻'
SET n:Time;

MATCH (n:Node) 
WHERE n.type = '人物'
SET n:Person;

MATCH (n:Node)
WHERE n.type = '地点' 
SET n:Location;

MATCH (n:Node)
WHERE n.type = '事件'
SET n:Event; 