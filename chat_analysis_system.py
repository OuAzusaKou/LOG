import os
import json
from datetime import datetime
from collections import defaultdict
from openai import OpenAI

from chat_analysis_to_embeddings import process_chat_analysis_to_embeddings
from date_data_translate import date_translate

from FlagEmbedding import BGEM3FlagModel

class ChatAnalysisSystem:
    def __init__(self, chat_folder, client):
        self.chat_folder = chat_folder
        self.client = client
        self.daily_records = {}
        self.overall_relationships = defaultdict(dict)

    def analyze_chats(self):
        for filename in os.listdir(self.chat_folder):
            if filename.endswith('.txt'):
                chat_content = self.read_chat_file(os.path.join(self.chat_folder, filename))
                self.analyze_chat_file(chat_content)
        
        self.summarize_overall_relationships()

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
                {"role": "system", "content": f"""你是一个专业的数据分析助手。需要根据'聊天记录'或者'故事'来分析出文本中出现的角色的：日期，今天和过去所在地点，事件，和角色的关系。\\
                 注意如果无法推测当天或者过去所在地点则为空。注意'聊天记录'或者'故事'中可能包含多个角色，时间，地点，事件，关系。注意'聊天记录'中的"你"，指"本人"，在回复中请用"本人"代替。
                 请用以下格式返回JSON,在非'聊天记录'中不存在'本人',请根据上下文尽量推断出年份，无法推断时请用空代替。: 
                 {{"本人"：[{{
                     "date": "2024-07-01",
                     "location": "北京",
                     "events": ["与朋友聚餐"],
                     "relationships": {{
                         "张三": {{"关系": "母子", "互动": "讨论工作"}},
                         "李四": {{"关系": "朋友", "互动": "一起吃饭"}}
                     }}
                 }},
                 {{
                     "date": "2024-06-01",
                     "location": "无锡",
                     "events": ["参加会议"],
                     "relationships": {{
                         "张三": {{"关系": "母子", "互动": "讨论工作"}},
                         "李四": {{"关系": "朋友", "互动": "一起吃饭"}}
                     }}
                 }}
                 ]，
                 "张三"：
                     [{{"date":"2024-07-01",
                     "location": "商场",
                     "events": ["参加会议"],
                     "relationships": {{
                         "你": {{"关系": "母子", "互动": "讨论工作"}},
                         "李四": {{"关系": "朋友", "互动": "一起吃饭"}}
                     }}}},
                    {{"date":"2024-03-02",
                     "location": "外面",
                     "events": ["参加会议"],
                     "relationships": {{
                         "你": {{"关系": "母子", "互动": "讨论工作"}},
                         "李四": {{"关系": "朋友", "互动": "一起吃饭"}}
                     }}}}，
                    ]
                 
                 }}"""
                 },

                {"role": "user", "content": f"""'聊天记录'或'故事'：{chat_content}"""}
            ],
            response_format={
                "type": "json_object"
            }
        )
        analysis_result = json.loads(analysis_result.choices[0].message.content)
        if self.daily_records.get(str(date)) is None:
            self.daily_records[str(date)] = [analysis_result]
        else:
            self.daily_records[str(date)].append(analysis_result)

        self.update_relationships(date, analysis_result)

    def update_relationships(self, date, daily_relationships):
        for per,dic in daily_relationships.items():
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

# 使用示例
if __name__ == '__main__':
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['HTTP_PROXY'] = 'http://192.168.224.76:7890'
    os.environ['HTTPS_PROXY'] = 'http://192.168.224.76:7890'

    client = OpenAI(
        api_key="sk-EjTppzE0xnr61QCj0d0MrZREohrwbV8xoMvOlvpw35g61vVG", # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://api.openai-proxy.org/v1",  # 填写DashScope服务的base_url
    )
    chat_folder = 'chat_record_file'
    analyzer = ChatAnalysisSystem(chat_folder, client)
    analyzer.analyze_chats()
    results = analyzer.get_results()
    
    # 保存聊天分析结果为JSON文件
    with open('chat_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
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
