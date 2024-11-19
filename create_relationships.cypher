// 创建关系
LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
MATCH (source:Node {id: row.source})
MATCH (target:Node {id: row.target}) 
WITH source, target, row
CALL {
    WITH source, target, row
    WITH source, target,
    CASE 
        WHEN row.relation IS NULL THEN 'RELATED_TO'
        ELSE row.relation
    END AS relType
    FOREACH(x IN CASE WHEN relType = 'RELATED_TO' THEN [1] ELSE [] END |
        MERGE (source)-[:RELATED_TO]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '父子' THEN [1] ELSE [] END |
        MERGE (source)-[:父子]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '母亲' THEN [1] ELSE [] END |
        MERGE (source)-[:母亲]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '丈夫' THEN [1] ELSE [] END |
        MERGE (source)-[:丈夫]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '妻子' THEN [1] ELSE [] END |
        MERGE (source)-[:妻子]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '恋人' THEN [1] ELSE [] END |
        MERGE (source)-[:恋人]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '女友' THEN [1] ELSE [] END |
        MERGE (source)-[:女友]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '父女' THEN [1] ELSE [] END |
        MERGE (source)-[:父女]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '亲生父母' THEN [1] ELSE [] END |
        MERGE (source)-[:亲生父母]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '儿子' THEN [1] ELSE [] END |
        MERGE (source)-[:儿子]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '父亲' THEN [1] ELSE [] END |
        MERGE (source)-[:父亲]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '朋友' THEN [1] ELSE [] END |
        MERGE (source)-[:朋友]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '合作伙伴' THEN [1] ELSE [] END |
        MERGE (source)-[:合作伙伴]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '家人' THEN [1] ELSE [] END |
        MERGE (source)-[:家人]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '建筑师' THEN [1] ELSE [] END |
        MERGE (source)-[:建筑师]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '家庭' THEN [1] ELSE [] END |
        MERGE (source)-[:家庭]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '学生' THEN [1] ELSE [] END |
        MERGE (source)-[:学生]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '保姆' THEN [1] ELSE [] END |
        MERGE (source)-[:保姆]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '同事' THEN [1] ELSE [] END |
        MERGE (source)-[:同事]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '工作' THEN [1] ELSE [] END |
        MERGE (source)-[:工作]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '邻居' THEN [1] ELSE [] END |
        MERGE (source)-[:邻居]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '工程师' THEN [1] ELSE [] END |
        MERGE (source)-[:工程师]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '合作创业伙伴' THEN [1] ELSE [] END |
        MERGE (source)-[:合作创业伙伴]->(target)
    )
    FOREACH(x IN CASE WHEN relType = '学校' THEN [1] ELSE [] END |
        MERGE (source)-[:学校]->(target)
    )
    RETURN count(*) AS cnt
}
RETURN count(*) AS total; 