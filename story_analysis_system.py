import os
import json
from datetime import datetime
from collections import defaultdict
import numpy as np
from openai import OpenAI

from chat_analysis_to_embeddings import process_chat_analysis_to_embeddings
from date_data_translate import date_translate

from FlagEmbedding import BGEM3FlagModel

class StoryAnalysisSystem:
    def __init__(self, chat_folder, client, encoder_model):
        self.chat_folder = chat_folder
        self.client = client
        self.daily_records = {}
        self.overall_relationships = defaultdict(dict)
        self.encoder_model = encoder_model
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

    def summarize_overall_relationships(self):
        # 这里可以添加更复杂的逻辑来总结整体关系
        pass

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


def setup_models():
    device = 'cuda:0'
    encoder_model = BGEM3FlagModel('./bge-m3', use_fp16=True)
    return encoder_model
# 使用示例
if __name__ == '__main__':
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['HTTP_PROXY'] = 'http://192.168.224.76:7890'
    os.environ['HTTPS_PROXY'] = 'http://192.168.224.76:7890'

    client = OpenAI(
        api_key="sk-EjTppzE0xnr61QCj0d0MrZREohrwbV8xoMvOlvpw35g61vVG", # 如果您没有配置环境变量，请��此处用您的API Key进行替换
        base_url="https://api.openai-proxy.org/v1",  # 填写DashScope服务的base_url
    )
    chat_folder = 'human_chart'
    encoder_model = setup_models()
    analyzer = StoryAnalysisSystem(chat_folder, client,encoder_model)
    # analyzer.analyze_chats()

    # results = analyzer.get_results()
    
    # # 保存聊天分析结果为JSON文件
    # with open('chat_analysis_results.json', 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)

    analyzer.process_and_save_embeddings('chat_analysis_results.json', './story_analysis_embedding_bge')
    
    # analyzer.check_analysis_result()

    
    # 执行日期翻译
    date_translate()
    
    # 处理聊天分析到嵌入
    json_path = 'sorted_chat_analysis_results.json'
    model_type = 'bgem3'  # 可以是 'bgem3' 或 'piccolo' 或 'bce'
    process_chat_analysis_to_embeddings(json_path, model_type)
    
    # 处理图片数据
    from image_processor import ImageProcessor
    processor = ImageProcessor(client)
    folder_path = 'img_record_file'
    image_data = processor.process_image_folder(folder_path)

    # 将图片数据保存为JSON文件
    with open('image_data.json', 'w', encoding='utf-8') as f:
        json.dump(image_data, f, ensure_ascii=False, indent=4)

    print("图片处理完成,结果已保存到 image_data.json")

    # 整合图片数据到聊天分析
    from integrate_image_data import integrate_image_data, load_json, save_json
    chat_data = load_json('chat_analysis_emb.json')
    encoder_model = BGEM3FlagModel('./bge-m3', use_fp16=True)
    embedding_folder = './chat_analysis_embedding_bge'
    os.makedirs(embedding_folder, exist_ok=True)
    updated_chat_data = integrate_image_data(chat_data, image_data, encoder_model, embedding_folder)
    save_json(updated_chat_data, 'updated_chat_analysis_emb.json')

    print(f"完成！修改后的JSON数据已保存到updated_chat_analysis_emb.json，图片描述的embedding已保存到{embedding_folder}文件夹")
