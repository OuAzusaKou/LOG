import os
import time
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from chat_analysis_system import ChatAnalysisSystem
from date_data_translate import load_json_data, sort_and_organize_data
from chat_analysis_to_embeddings import process_chat_analysis_to_embeddings
from openai import OpenAI
import ssl

class ChatFileHandler(FileSystemEventHandler):
    def __init__(self, chat_folder, client):
        self.chat_folder = chat_folder
        self.client = client
        self.last_processed_time = self.get_last_processed_time()
        self.analyzer = ChatAnalysisSystem(self.chat_folder, self.client)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            print(f"检测到新文件：{event.src_path}")
            self.process_updates(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            current_time = time.time()
            if current_time - self.last_processed_time > 60:  # 每分钟最多处理一次
                print(f"检测到文件变化：{event.src_path}")
                self.process_updates(event.src_path)
                self.last_processed_time = current_time

    def process_updates(self, new_chat_file):
        print(f"开始处理文件：{new_chat_file}")
        # 1. 运行 incremental_update
        self.analyzer.incremental_update(new_chat_file)
        
        # 2. 获取更新后的结果
        input_file = 'chat_analysis_results.json'
    
        data = load_json_data(input_file)
        sorted_data = sort_and_organize_data(data)
        # with open('chat_analysis_results.json', 'w', encoding='utf-8') as f:
        #     json.dump(results, f, ensure_ascii=False, indent=2)

        # 3. 运行 date_data_translate
        # sorted_data = sort_and_organize_data(results)
        with open('sorted_chat_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, ensure_ascii=False, indent=2)
        
        # 4. 运行 chat_analysis_to_embeddings
        process_chat_analysis_to_embeddings('sorted_chat_analysis_results.json', 'bgem3')
        
        print(f"文件 {new_chat_file} 处理完成")

    def get_last_processed_time(self):
        return time.time()

def main():
    chat_folder = 'chat_record_file'
    
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['HTTP_PROXY'] = 'http://192.168.224.76:7890'
    os.environ['HTTPS_PROXY'] = 'http://192.168.224.76:7890'

    client = OpenAI(
        api_key="sk-EjTppzE0xnr61QCj0d0MrZREohrwbV8xoMvOlvpw35g61vVG",
        base_url="https://api.openai-proxy.org/v1",
    )

    event_handler = ChatFileHandler(chat_folder, client)
    observer = Observer()
    observer.schedule(event_handler, chat_folder, recursive=False)
    observer.start()

    try:
        print(f"开始监控 {chat_folder} 文件夹...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
