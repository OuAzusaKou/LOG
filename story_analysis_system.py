import os
import json
from datetime import datetime
from collections import defaultdict
from neo4j import GraphDatabase
import numpy as np
from openai import OpenAI

from chat_analysis_to_embeddings import process_chat_analysis_to_embeddings
from data_input.video_input import read_video_segment
from date_data_translate import date_translate

from FlagEmbedding import BGEM3FlagModel

from qdrant_client import QdrantClient
from qdrant_client.http import models

from image_embedding.split_video import detect_scene_changes, extract_key_frames, save_key_frames

class StoryAnalysisSystem:
    def __init__(self, chat_folder, client, encoder_model, neo4j_driver, collection_name='story_analysis'):
        self.chat_folder = chat_folder
        self.client = client
        self.daily_records = {}
        self.overall_relationships = defaultdict(dict)
        self.encoder_model = encoder_model
        
        
        # 使用story_analysis数据库连接
        self.driver = neo4j_driver
        
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient("localhost", port=6333)
        self.init_qdrant_collection()
    def init_qdrant_collection(self):
        collections = self.qdrant_client.get_collections().collections
        existing_collections = [collection.name for collection in collections]

        if self.collection_name not in existing_collections:
            print(f"创建新集合: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024,  # 向量维度
                    distance=models.Distance.COSINE  # 距离度量方式
                )
            )
        else:
            print(f"集合 {self.collection_name} 已存在，跳过创建步骤")

    def reset_all_database(self):
        self.qdrant_client.delete_collection(self.collection_name)
        self.init_qdrant_collection()
        self.driver.execute_query("MATCH (n) DETACH DELETE n")

    def analyze_chats(self):
        for filename in os.listdir(self.chat_folder):
            if filename.endswith('.txt'):
                chat_content = self.read_chat_file(os.path.join(self.chat_folder, filename))
                self.analyze_chat_file(chat_content)
        
        self.summarize_overall_relationships()

    def check_analysis_result(self):
        for chapter, records in self.daily_records.items():
            for record in records:
                # 将每个记录发送到AI接口进行检测和修正
                corrected_record = self.correct_record_with_ai(chapter, record)
                # 更新记录
                record.update(corrected_record)
        return self.daily_records

    def correct_record_with_ai(self, chapter, record):
        # 获取原文内容
        original_text = self.get_original_text(chapter)
        # 去掉 '原文' 字段
        record_without_original = {k: v for k, v in record.items() if k != '原文'}

        other_records = self.get_non_original_records()
        # 调用AI接口进行检测和修正
        corrected_record = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "请根据以下原文和其他相关信息修正'原始记录'中人物姓名和关系的一致性问题。人物希望使用全名，注意使用和'原始记录'中一样的JSON格式回答，请使用中文。"},
                {"role": "user", "content": f"原文：{original_text}\n原始记录：{json.dumps(record_without_original, ensure_ascii=False)}\n其他相关信息：{json.dumps(other_records, ensure_ascii=False)}\n"}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(corrected_record.choices[0].message.content)

    def get_original_text(self, chapter):
        # 从 chat_analysis_results.json 中获取相应章节的原文
        with open('chat_analysis_results.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        for chapter_key, records in data['daily_records'].items():
            if chapter_key == chapter:
                return records[0]['原文']
        return ""

    def read_chat_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def analyze_chat_file(self, chat_content):
        # 将聊天内容按日期分割
        daily_chats = self.split_chat_by_date(chat_content)
        for date, daily_content in daily_chats.items():
            self.analyze_daily_chat(date, daily_content)

    def split_chat_by_date(self, chat_content):
        daily_chats = {}
        current_date = None
        current_content = []

        for line in chat_content.split('\n'):
            if line.startswith('时间戳：'):
                date_str = line.split('：')[1]
                new_date = self.parse_date_or_chapter(date_str)
                if new_date != current_date:
                    if current_date:
                        daily_chats[current_date] = '\n'.join(current_content)
                    current_date = new_date
                    current_content = []
            current_content.append(line)

        if current_date:
            daily_chats[current_date] = '\n'.join(current_content)

        return daily_chats

    def parse_date_or_chapter(self, date_str):
        if date_str.startswith("第"):
            return f"章节标识：{date_str}"
        else:
            # 尝试解析为日期
            date_str = date_str.split()[0]
            new_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            return new_date

    def analyze_daily_chat(self, date, chat_content):
        analysis_result = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""你是一个专业的'故事'数据分析助手。需要根据'故事'来分析出文本中出现的角色的：日期，现在和过去所在地点，事件，和角色的关系。\\
                 注意如果无法推测现在或者过去所在地点则为空。注意'故事'中可能包含多个角色，时间，地点，事件，关系。不要出现\'\\\'符号
                 请用严格按照以下格式返回JSON,严格包含'date','location','events','relationships'四个键值对，请根据上下文尽量推断出年份，无法推断时用空代替。: 
                 {{"阿里巴巴"：[{{
                     "date": "2024-07-01",
                     "location": "北京",
                     "events": ["在北京创立阿里巴巴"],
                     "relationships": {{
                         "张三": {{"关系": "母子", "互动": "讨论工作"}},
                         "李四": {{"关系": "朋友", "互动": "一起吃饭"}}
                     }}
                 }},
                 {{
                     "date": "2024-06-01",
                     "location": "无锡",
                     "events": ["去旅行，吃了醉蟹，吃了西湖醋鱼，和朋友拍下了决定历史的照片"],
                     "relationships": {{
                         "张三": {{"关系": "母子", "互动": "讨论工作"}},
                         "李四": {{"关系": "朋友", "互动": "一起吃饭"}}
                     }}
                 }}
                 ]，
                 "张三"：[{{"date":"2024-07-01",
                     "location": "商场",
                     "events": ["与同事开会，并记录了决定历史的一刻"],
                     "relationships": {{
                         "李四": {{"关系": "朋友", "互动": "一起吃饭"}}
                     }}}},
                    {{"date":"2024-03-02",
                     "location": "外面",
                     "events": ["逛了逛商场，买了个手机，买了个电脑，买了个平板"],
                     "relationships": {{
                         "李四": {{"关系": "朋友", "互动": "一起吃饭"}}
                     }}}}，
                    ],
                 "李四"：[{{"date":"2024-07-01",
                     "location": "电话亭",
                     "events": ["打电话给朋友约饭，并拍下了决定历史的照片"],
                     "relationships": {{
                         "张三": {{"关系": "朋友", "互动": "一起吃饭"}}
                     }}}},
                    {{"date":"2024-03-02",
                     "location": "南京",
                     "events": ["找了个旅馆，美美得睡了一觉"],
                     "relationships": {{
                         "阿里巴巴": {{"关系": "朋友", "互动": "一起吃饭"}}
                     }}}}，
                    ]
                 
                 }}"""
                 },

                {"role": "user", "content": f"""'故事'：{chat_content}"""}
            ],
            response_format={
                "type": "json_object"
            }
        )
        analysis_result = json.loads(analysis_result.choices[0].message.content)
        analysis_result['原文'] = chat_content
        if self.daily_records.get(str(date)) is None:
            self.daily_records[str(date)] = [analysis_result]
        else:
            self.daily_records[str(date)].append(analysis_result)

        self.update_relationships(date, analysis_result)

    def update_relationships(self, date, daily_relationships):
        for per,dic in daily_relationships.items():
            if per != '原文':
                if dic:
                    for i,date_info in enumerate(dic):
                        for person, info in date_info['relationships'].items():
                            if ((per+'_'+person) not in self.overall_relationships) and ((person+'_'+ per) not in self.overall_relationships):
                                self.overall_relationships[per+'_'+person] = info
                        # else:
                        #     # 这里可以添加更复杂的逻辑来更新关系
                        #     self.overall_relationships[per+'_'+person]['最近互动'] = f"{date}: {info['互动']}"


    def get_results(self):
        return {
            'daily_records': self.daily_records,
            'overall_relationships': dict(self.overall_relationships)
        }

    def incremental_update(self, new_chat_file):
        # 读取现有的分析结果
        try:
            with open('chat_analysis_results.json', 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except FileNotFoundError:
            existing_results = {'daily_records': {}, 'overall_relationships': {}}

        # 分析新的聊天文件
        chat_content = self.read_chat_file(new_chat_file)
        daily_chats = self.split_chat_by_date(chat_content)

        for date, daily_content in daily_chats.items():
            self.analyze_daily_chat(date, daily_content)

            # 更新或添加到现有结果
            date_str = str(date)
            if date_str in existing_results['daily_records']:
                existing_results['daily_records'][date_str].extend(self.daily_records[date_str])
            else:
                existing_results['daily_records'][date_str] = self.daily_records[date_str]

        # 更新整体关系
        for person, info in self.overall_relationships.items():
            if person in existing_results['overall_relationships']:
                existing_results['overall_relationships'][person].update(info)
            else:
                existing_results['overall_relationships'][person] = info

        # 保存更新后的结果
        with open('chat_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)

        print(f"已成功更新 chat_analysis_results.json 文件")

    def get_non_original_records(self):
        non_original_records = []
        for date, records in self.daily_records.items():
            non_original_records.append(
                [{k: v for k, v in record.items() if k != '原文'} for record in records]
            )
        return non_original_records
    

    
    def process_and_save_embeddings(self, json_path, embedding_folder):
        # 确保embedding文件夹存在
        os.makedirs(embedding_folder, exist_ok=True)

        # 读取 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        def get_embedding(text):
            return self.encoder_model.encode(text)['dense_vecs']

        # 函数：递归遍历 JSON 数据并添加 `embeddings` 字段
        def add_embeddings(node, date, person_index):
            if isinstance(node, dict):
                node['embeddings'] = []

                # 处理日期的embedding
                if 'date' in node:
                    date_address = f"date_{node['date']}_{person_index}"
                    node['embeddings'].append(date_address)
                    date_embeddings = get_embedding(node['date'])
                    np.save(os.path.join(embedding_folder, date_address), date_embeddings)

                # 处理人物的embedding
                if 'person' in node:
                    person_address = f"person_{node['person']}"
                    node['embeddings'].append(person_address)
                    person_embeddings = get_embedding(node['person'])
                    np.save(os.path.join(embedding_folder, person_address), person_embeddings)

                # 处理地点的embedding
                if 'location' in node:
                    location_address = f"location_{node['location']}"
                    node['embeddings'].append(location_address)
                    location_embeddings = get_embedding(node['location'])
                    np.save(os.path.join(embedding_folder, location_address), location_embeddings)
                    
                if 'events' in node:
                    for event in node['events']:
                        event_address = f"event_{event}"
                        node['embeddings'].append(event_address)
                        event_embeddings = get_embedding(event)
                        np.save(os.path.join(embedding_folder, event_address), event_embeddings)
                # 处理关系的embedding
                if 'relationships' in node:
                    for rel_person, rel_info in node['relationships'].items():
                        rel_address = f"relation_{rel_info['关系']}"
                        node['embeddings'].append(rel_address)
                        rel_embeddings = get_embedding(rel_info['关系'])
                        np.save(os.path.join(embedding_folder, rel_address), rel_embeddings)

        # 遍历数据并添加embeddings
        for date, people_list in data['daily_records'].items():
            for index, person_data in enumerate(people_list):
                for person, details in person_data.items():
                    if person != '原文':   
                        for detail in details:
                            detail['person'] = person  # 添加person信息
                            add_embeddings(detail, date, index)

        # 处理overall_relationships
        if 'overall_relationships' in data:
            data['overall_relationships']['embeddings'] = []
            for person, rel_info in data['overall_relationships'].items():
                if person != 'embeddings':  # 跳过embeddings键
                    rel_address = f"overall_relation_{person}"
                    data['overall_relationships']['embeddings'].append(rel_address)
                    rel_embeddings = get_embedding(rel_info['关系'])
                    np.save(os.path.join(embedding_folder, rel_address), rel_embeddings)

        # 将修改后的 JSON 数据写回文件
        with open('chat_analysis_emb.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"完成！修改后的 JSON 数据已保存到 {json_path}，embedding 已保存到 {embedding_folder} 文件夹")
    
    def check_similar_vectors(self, vector, collection_name):
        search_result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=1024,
            score_threshold=0.9
        )
        return search_result[0] if search_result else None

    def add_graph_embedding_to_qdrant(self,graph_address,embeddings):
        node_match = self.check_similar_vectors(embeddings, self.collection_name)
        if not node_match:
            # 使用绝对值确保ID为正数
            point_id = abs(hash(f"graph_{graph_address}")) % (2**63 - 1)  # 确保ID在64位无符号整数范围内
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=[point_id],  # 添加正整数ID
                    vectors=[embeddings],
                    payloads=[{"type": 'graph', "value": graph_address}]
                )
            )
            return graph_address, embeddings
        else:
            return node_match.payload['value'], embeddings
        

    def get_embedding_and_add_to_qdrant(self, node_type, content):
        node_embeddings = self.encoder_model.encode(content)['dense_vecs']
        node_match = self.check_similar_vectors(node_embeddings, self.collection_name)
        if not node_match:
            # 使用绝对值确保ID为正数
            point_id = abs(hash(f"{node_type}_{content}")) % (2**63 - 1)  # 确保ID在64位无符号整数范围内
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=[point_id],  # 添加正整数ID
                    vectors=[node_embeddings],
                    payloads=[{"type": node_type, "value": content}]
                )
            )
            return content, node_embeddings
        else:
            return node_match.payload['value'], node_embeddings

    def entity_relation_extract(self,chat_content):
        analysis_result = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""你是一个专业的'故事'数据分析助手。需根据'故事'来分析出文本中出现的角色的：日期，现在和过去所在地点，事件，和角色的关系。\\
                    注意如果无法推测现在或者过去所在地点则为空。注意'故事'中可能包含多个角色，时间，地点，事件，关系。不要出现\'\\\'符号
                    请用严格按照以下格式返回JSON,严格包含'date','location','events','relationships'四个键值对，请根据上下文尽量推断出年份，无法推断时请用空代替。: 
                    {{"阿里巴巴"：[{{
                        "date": "2024-07-01",
                        "location": "北京",
                        "events": ["在北京创立阿里巴巴"],
                        "relationships": {{
                            "张三": {{"关系": "母子", "互动": "讨论工作"}},
                            "李四": {{"关系": "朋友", "互动": "一起吃饭"}}
                        }}
                    }},
                    {{
                        "date": "2024-06-01",
                        "location": "无锡",
                        "events": ["去旅行，吃了醉蟹，吃了西湖醋鱼，和朋友拍下了决定历史的照片"],
                        "relationships": {{
                            "张三": {{"关系": "母子", "互动": "讨论工作"}},
                            "李四": {{"关系": "朋友", "互动": "一起吃饭"}}
                        }}
                    }}
                    ]，
                    "张三"：[{{"date":"2024-07-01",
                        "location": "商场",
                        "events": ["与同事开会，并记录了决定历史的一刻"],
                        "relationships": {{
                            "李四": {{"关系": "朋友", "互动": "一起吃饭"}}
                        }}}},
                    {{"date":"2024-03-02",
                        "location": "外面",
                        "events": ["逛了逛商场，买了个手机，买了个电脑，买了个平板"],
                        "relationships": {{
                            "李四": {{"关系": "朋友", "互动": "一起吃饭"}}
                        }}}}，
                    ],
                    "李四"：[{{"date":"2024-07-01",
                        "location": "电话亭",
                        "events": ["打电话给朋友约饭，并拍下了决定历史的照片"],
                        "relationships": {{
                            "张三": {{"关系": "朋友", "互动": "一起吃饭"}}
                        }}}},
                    {{"date":"2024-03-02",
                        "location": "南京",
                        "events": ["找了个旅馆，美美得睡了一觉"],
                        "relationships": {{
                            "阿里巴巴": {{"关系": "朋友", "互动": "一起吃饭"}}
                        }}}}，
                    ]
                    
                    }}"""
                    },

                {"role": "user", "content": f"""'故事'：{chat_content}"""}
            ],
            response_format={
                "type": "json_object"
            }
        )
        analysis_result = json.loads(analysis_result.choices[0].message.content)
        return analysis_result
    
    def add_entity_to_neo4j(self,entity_name_1,entity_name_2,entity_type_1,entity_type_2,relation_type):
        with self.driver.session() as session:

            # 创建节点和关系的Cypher查询保持不变
            create_nodes_query = f"""
            MERGE (n:Node:{entity_type_1} {id: $id_1})
            MERGE (m:Node:{entity_type_2} {id: $id_2})
            MERGE (n)-[:{relation_type}]-(m)
            RETURN n, m
            """
            
            session.run(create_nodes_query, 
                    id_1=entity_name_1,
                    id_2=entity_name_2)
    
    def add_content_to_database(self,chat_content):

        analysis_result = self.entity_relation_extract(chat_content)
        

        content_match,content_embeddings = self.get_embedding_and_add_to_qdrant('content',chat_content)

        for person, records in analysis_result.items():

            person_match,person_embeddings = self.get_embedding_and_add_to_qdrant('person',person)

            for record in records:
                location = record['location']
                events = record['events']
                relationships = record['relationships']
                

                
                location_match,location_embeddings = self.get_embedding_and_add_to_qdrant('location',location)

                events_embeddings_list = []
                events_match_list = []
                for event in events:
                    event_match,event_embeddings = self.get_embedding_and_add_to_qdrant('event',event)
                    events_embeddings_list.append(event_embeddings)
                    events_match_list.append(event_match)

                date_match,date_embeddings = self.get_embedding_and_add_to_qdrant('date',record['date'])

                relationships_embeddings_list = []
                per_embeddings_list = []
                per_match_list = []
                relationships_match_list = []
                for per,rel_info in relationships.items():
                    rel_match,rel_embeddings = self.get_embedding_and_add_to_qdrant('relation',rel_info['关系'])
                    relationships_embeddings_list.append(rel_embeddings)
                    relationships_match_list.append(rel_match)
                    per_match,per_embeddings = self.get_embedding_and_add_to_qdrant('person',per)
                    per_embeddings_list.append(per_embeddings)
                    per_match_list.append(per_match)
                # Neo4j数据库操作
                with self.driver.session() as session:
                    # 使用实际的或匹配到的节点ID
                    location_id = location_match
                    person_id = person_match
                    date_id = date_match

                    # 创建节点和关系的Cypher查询保持不变
                    create_nodes_query = """
                    MERGE (n:Node:Location {id: $id_1})
                    MERGE (m:Node:Person {id: $id_2})
                    MERGE (n)-[:地点]-(m)
                    RETURN n, m
                    """
                    
                    session.run(create_nodes_query, 
                            id_1=location_id,
                            id_2=person_id)
                    
                    create_nodes_query = """
                    MERGE (n:Node:Location {id: $id_1})
                    MERGE (m:Node:Time {id: $id_2})
                    MERGE (n)-[:时刻]-(m)
                    RETURN n, m
                    """
                    
                    session.run(create_nodes_query, 
                            id_1=location_id,
                            id_2=date_id)
                    
                    create_nodes_query = """
                    MERGE (n:Node:Person {id: $id_1})
                    MERGE (m:Node:Time {id: $id_2})
                    MERGE (n)-[:时刻]-(m)
                    RETURN n, m
                    """
                    
                    session.run(create_nodes_query, 
                            id_1=person_id,
                            id_2=date_id)
                    
                    for event_id in events_match_list:
                        create_nodes_query = """
                        MERGE (n:Node:Person {id: $id_1})
                        MERGE (m:Node:Event {id: $id_2})
                        MERGE (n)-[:事件]-(m)
                        RETURN n, m
                        """
                        session.run(create_nodes_query, 
                            id_1=person_id,
                            id_2=event_id)
                    for rel_person, relation_id in zip(per_match_list,relationships_match_list):
                        create_nodes_query = """
                        MERGE (n:Node:Person {id: $id_1})
                        MERGE (m:Node:Person {id: $id_2})
                        MERGE (n)-[r:RELATIONSHIP {type: $rel_type}]-(m)
                        RETURN n, m
                        """
                        session.run(create_nodes_query, 
                            id_1=person_id,
                            id_2=rel_person,
                            rel_type=relation_id)
                        
                    content_id = content_match
                    create_nodes_query = """
                    MERGE (n:Node:Content {id: $id_1})
                    MERGE (m:Node:Time {id: $id_2})
                    MERGE (n)-[:原文]-(m)
                    RETURN n, m
                    """
                    session.run(create_nodes_query, 
                            id_1=content_id,
                            id_2=date_id)
                    

        analysis_result['原文'] = chat_content

                    
        # if self.daily_records.get(str(date)) is None:
        #     self.daily_records[str(date)] = [analysis_result]
        # else:
        #     self.daily_records[str(date)].append(analysis_result)



    def query_embedding_match(self,query_vector,top_k=5,score_threshold=0.5):
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold
        )
        #根据返回向量结果找到相应的neo4j节点
        node_ids = [result.payload['value'] for result in search_result]
        return search_result,node_ids
    
    def bfs_k_steps_with_paths(self, start_node, k):
        """
        使用BFS算法在Neo4j中搜索从起始节点开始k步内可达的所有节点和路径
        
        Args:
            driver: Neo4j驱动
            start_node: 起始节���ID
            k: 最大步数
        
        Returns:
            包含所有路径的列表
        """
        query = """
        MATCH path = (start:Node {id: $start_id})-[*1..{k}]-(end:Node)
        WHERE start <> end
        RETURN path
        """
        
        with self.driver.session() as session:
            result = session.run(query, start_id=start_node, k=k)
            paths = []
            
            for record in result:
                path = record['path']
                path_info = {
                    'nodes': [],
                    'relationships': []
                }
                
                # 提取路径中的节点信息
                for node in path.nodes:
                    node_info = {
                        'id': node['id'],
                        'labels': list(node.labels)
                    }
                    path_info['nodes'].append(node_info)
                
                # 提取路径中的关系信息
                for rel in path.relationships:
                    rel_info = {
                        'type': rel.type,
                        'start_node': rel.start_node['id'],
                        'end_node': rel.end_node['id']
                    }
                    if 'type' in rel:  # 如果关系有type属性
                        rel_info['relation_type'] = rel['type']
                    path_info['relationships'].append(rel_info)
                
                paths.append(path_info)
                
            return paths

    def bfs_k_steps(self, start_nodes, k):
        """
        对多个起始节点执行k步BFS搜索
        
        Args:
            start_nodes: 起始节点ID列表
            k: 最大步数
        
        Returns:
            包含所有路径的字典，以起始节点为键
        """
        results = {}
        for start_node in start_nodes:
            paths = self.bfs_k_steps_with_paths(start_node, k)
            results[start_node] = paths
            
            # 打印搜索结果的摘要
            print(f"\n从节点 {start_node} 开始的 {k} 步搜索结果:")
            print(f"找到 {len(paths)} 条路径")
            for i, path in enumerate(paths, 1):
                print(f"\n路径 {i}:")
                print(f"节点: {' -> '.join([node['id'] for node in path['nodes']])}")
                print(f"关系: {' -> '.join([rel['type'] for rel in path['relationships']])}")
        
        return results
    
    def qwen_image(self, image_path):
        # 读取图片文件为base64格式
        import base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 使用OpenAI的vision模型进行图片理解
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # 使用支持图像的模型
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    
                        {
                            "type": "text", 
                            "text": "请分析这张图片的内容。"
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        try:
            # 解析返回的JSON结果
            result = response.choices[0].message.content
            return result
        except json.JSONDecodeError:
            # 如果返回结果不是有效的JSON格式，返回原始文本
            return {"error": "无法解析图片分析结果", "raw_response": response.choices[0].message.content}


def setup_models():
    device = 'cuda:0'
    encoder_model = BGEM3FlagModel('./bge-m3', use_fp16=True)
    return encoder_model
# 使用示例
if __name__ == '__main__':
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['HTTP_PROXY'] = 'http://192.168.224.76:8960'
    os.environ['HTTPS_PROXY'] = 'http://192.168.224.76:8960'
    # os.environ['HTTP_PROXY'] = 'http://192.168.11.166:7890'
    # os.environ['HTTPS_PROXY'] = 'http://192.168.11.166:7890'
    os.environ['NO_PROXY'] = 'localhost'
    
    client = OpenAI(
        api_key="sk-EjTppzE0xnr61QCj0d0MrZREohrwbV8xoMvOlvpw35g61vVG", # 如果您没有配置环境变量，请此处用您的API Key进行替换
        base_url="https://api.openai-proxy.org/v1",  # 填写DashScope服务的base_url
    )
    
    uri = "bolt://localhost:7687"  # Neo4j默认地址和端口
    user = "neo4j"                 # 默认用户名
    password = "12345678"     # 你的密码

    # 首先创建一个默认的驱动实例
    driver = GraphDatabase.driver(uri, database='story', auth=(user, password))

    chat_folder = 'human_chart'
    encoder_model = setup_models()
    analyzer = StoryAnalysisSystem(chat_folder, client,encoder_model,driver)
    analyzer.reset_all_database()

    frames, fps = read_video_segment(
        video_path="./data_input/agan.mp4",
        start_time=2.0,
        end_time=10.0
    )
    
    scene_changes = detect_scene_changes(frames,0.99999)

    key_frames = extract_key_frames(frames, scene_changes)

    save_key_frames(key_frames, './key_frames')

    for root, dirs, files in os.walk('./key_frames'):
        for file in files:
            result_img = analyzer.qwen_image(os.path.join(root, file))

            analyzer.add_content_to_database(result_img)

            print(result_img)


    #



    # print(scene_changes)

    # result_img = analyzer.qwen_image('./img_record_file/1644895299_-1599608410_660496424.jpg')
    # print(result_img)
    # analyzer.add_content_to_database('阿里巴巴和马云在东京打败了四十大盗。')
    # analyzer.add_content_to_database('哈利波特和马云在伦敦坐火车喝茶。')
    




    # analyzer.analyze_chats()

    # results = analyzer.get_results()
    
    # # 保存聊天分析结果为JSON文件
    # with open('chat_analysis_results.json', 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)

    # analyzer.process_and_save_embeddings('chat_analysis_results.json', './story_analysis_embedding_bge')
    
    # analyzer.check_analysis_result()

    
    # # 执行日期翻译
    # date_translate()
    
    # # 处理聊天分析到嵌入
    # json_path = 'sorted_chat_analysis_results.json'
    # model_type = 'bgem3'  # 可以是 'bgem3' 或 'piccolo' 或 'bce'
    # process_chat_analysis_to_embeddings(json_path, model_type)
    
    # 处理图片数据
    # from image_processor import ImageProcessor
    # processor = ImageProcessor(client)
    # folder_path = 'img_record_file'
    # image_data = processor.process_image_folder(folder_path)

    # # 将图片数据保存为JSON文件
    # with open('image_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(image_data, f, ensure_ascii=False, indent=4)

    # print("图片处理完成,结果已保存到 image_data.json")

    # # 整合图片数据到聊天分析
    # from integrate_image_data import integrate_image_data, load_json, save_json
    # chat_data = load_json('chat_analysis_emb.json')
    # encoder_model = BGEM3FlagModel('./bge-m3', use_fp16=True)
    # embedding_folder = './chat_analysis_embedding_bge'
    # os.makedirs(embedding_folder, exist_ok=True)
    # updated_chat_data = integrate_image_data(chat_data, image_data, encoder_model, embedding_folder)
    # save_json(updated_chat_data, 'updated_chat_analysis_emb.json')

    # print(f"完成！修改后的JSON数据已保存到updated_chat_analysis_emb.json，图片描述的embedding已保存到{embedding_folder}文件夹")
