from collections import defaultdict
import os
from flask import json
import openai
from FlagEmbedding import BGEM3FlagModel
import numpy as np

class QueryAnalysisSystem:
    def __init__(self, client, encoder_model, chat_folder='human_chart'):
        self.chat_folder = chat_folder
        self.client = client
        self.encoder_model = encoder_model
        self.daily_records = {}
        self.overall_relationships = defaultdict(dict)

    def extract_entities_and_relationships(self, question_text):
        analysis_result = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""你是一个专业的'问题'数据分析助手。需要根据'问题'来分析出文本中出现的实体和关系，以及回答问题可能需要用到的实体和关系。\\
                 请用严格按照以下格式返回JSON,严格包含'entities','relationships'两个键值对。:
                 {{"entities": ["实体1", "实体2", "实体3"],
                   "relationships": [
                       {{"entity1": "实体1", "entity2": "实体2", "relationship": "关系描述"}},
                       {{"entity1": "实体2", "entity2": "实体3", "relationship": "关系描述"}}
                   ]
                 }}
                 以下是问题答案示例：
                 问题：红孩儿的父亲和铁扇公主的母亲是什么关系
                 答案：{{
                        "entities": ["红孩儿", “红孩儿的父亲”，"铁扇公主"，“铁扇公主的母亲”],
                        "relationships": [
                            {{"entity1": "红孩儿的父亲", "entity2": "铁扇公主的母亲", "relationship": "未知"}},
                            {{"entity1": "红孩儿", "entity2": "铁扇公主", "relationship": "未知"}},
                            {{"entity1": "红孩儿的父亲", "entity2": "红孩儿", "relationship": "父子"}},
                            {{"entity1": "铁扇公主的母亲", "entity2": "铁扇公主", "relationship": "母女"}}
                        ]
                    }}，
                 问题：阿里巴巴的孩子和四十大盗的村子中 的村民是什么关系
                 答案：{{
                        "entities": ["阿里巴巴", "阿里巴巴的孩子", "四十大盗", "四十大盗的村子", "村民"],
                        "relationships": [
                            {{"entity1": "阿里巴巴的孩子", "entity2": "四十大盗的村子", "relationship": "未知"}},
                            {{"entity1": "四十大盗的村子", "entity2": "村民", "relationship": "居住"}}，
                            {{"entity1": "阿里巴巴", "entity2": "阿里巴巴的孩子", "relationship": "父子"}}，
                            {{"entity1": "四十大盗", "entity2": "四十大盗的村子", "relationship": "居住"}}，
                            {{"entity1": "四十大盗", "entity2": "村民", "relationship": "村民"}}
                        ]
                    }}
                 """
                },
                {"role": "user", "content": f"""'问题'：{question_text}"""}
            ],
            response_format={
                "type": "json_object"
            }
        )
        return analysis_result.choices[0].message.content
    
    def get_embeddings(self, analysis_result):
        entities = analysis_result['entities']
        relationships = analysis_result['relationships']

        # 获取实体的嵌入
        entity_embeddings = {entity: self.encoder_model.encode(entity)['dense_vecs'] for entity in entities}

        # 获取关系的嵌入
        relationship_embeddings = {rel['relationship']: self.encoder_model.encode(f"{rel['relationship']}")['dense_vecs'] for rel in relationships}

        return entity_embeddings, relationship_embeddings


def setup_models():
    device = 'cuda:0'
    encoder_model = BGEM3FlagModel('./bge-m3', use_fp16=True)
    return encoder_model

if __name__ == '__main__':
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['HTTP_PROXY'] = 'http://192.168.224.76:7890'
    os.environ['HTTPS_PROXY'] = 'http://192.168.224.76:7890'

    client = openai.OpenAI(
        api_key="sk-EjTppzE0xnr61QCj0d0MrZREohrwbV8xoMvOlvpw35g61vVG", # 如果您没有配置环境变量，请此处用您的API Key进行替换
        base_url="https://api.openai-proxy.org/v1",  # 填写DashScope服务的base_url
    )

    # 初始化 bge-m3 模型的编码器
    encoder_model = setup_models()

    query_analysis_system = QueryAnalysisSystem(client=client, encoder_model=encoder_model, chat_folder='human_chart')
    while True:
        question = input("请输入问题：")
        analysis_result = query_analysis_system.extract_entities_and_relationships(question)
        analysis_result = json.loads(analysis_result)
        entity_embeddings, relationship_embeddings = query_analysis_system.get_embeddings(analysis_result)
        print("实体的嵌入：", entity_embeddings)
        print("关系的嵌入：", relationship_embeddings)
