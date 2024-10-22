import os
import time
import json
import numpy as np
from multiprocessing import Process, Queue
from docx2md import convert_docx_to_md, convert_pdf_to_md
from json2embeddings import process_json_to_embeddings
from response_agent import get_agent_response
from title_toc import process_md_files
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from sentence_transformers import SentenceTransformer
from BCEmbedding import EmbeddingModel
from qwen_agent.agents import Assistant
from json2adjmatrix import json2adj, load_json
from query2ac_node import get_context_by_node, index_by_number, querypath2node

import signal
import sys

from openai import OpenAI
import os

def get_response(question, contexts):
    client = OpenAI(
        api_key="sk-EjTppzE0xnr61QCj0d0MrZREohrwbV8xoMvOlvpw35g61vVG", # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://api.openai-proxy.org/v1",  # 填写DashScope服务的base_url
    )
    completion = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        model = "gpt-4o",
        messages=[
            {'role': 'system', 'content': '''你是一个专门回答问题的助手，需要结合"context"来回答"query"
        始终用中文回复。'''},
            {'role': 'user', 'content': str({'context': contexts, 'query': question})}]
        )
    # print(completion.model_dump_json())
    return completion.choices[0].message.content


def process_documents(directory_path, out_md_directory, json_path, model_type, output_json_path, process_docx=True, process_pdf=True):
    # 创建输出目录
    os.makedirs(out_md_directory, exist_ok=True)
    
    if process_docx:
        # 转换 docx 文件
        convert_docx_to_md(directory_path, out_md_directory)
    
    if process_pdf:
        # 转换 pdf 文件
        convert_pdf_to_md(directory_path, out_md_directory)
    
    # 处理 md 文件
    process_md_files(out_md_directory, json_path)
    
    # 处理 json 到 embeddings
    process_json_to_embeddings(json_path, model_type, output_json_path)

def setup_models(model_type,local_model=False,n_jump=False):
    device = 'cuda:0'

    # 配置embedding模型
    if model_type == 'bgem3':
        encoder_model = BGEM3FlagModel('./bge-m3', use_fp16=True)
    elif model_type == 'piccolo':
        encoder_model = SentenceTransformer('./snn_rag/piccolo').to(device)
    elif model_type == 'bce':
        encoder_model = EmbeddingModel(model_name_or_path="./bce-base")
    else:
        raise ValueError("不支持的模型类型")

    # 配置reranker模型
    if n_jump:
        reranker = FlagReranker('./bge_reranker_large', use_fp16=True)
    else:
        reranker = None
    if local_model:
    # 配置LLM API
        llm_cfg = {
            'model': 'Qwen2-7B-Instruct',
            'model_server': 'http://0.0.0.0:8000/v1/',
            'api_key': 'EMPTY',
            'generate_cfg': {
                'top_p': 0.8,
                "max_input_tokens": 32768
            }
        }

        system_instruction = '''你是一个专门回答问题的助手，需要结合"context"来回答"query"
        始终用中文回复。'''
        tools = []
        files = []
        bot = Assistant(llm=llm_cfg,
                        system_message=system_instruction,
                        function_list=tools,
                        files=files)
    else:
        bot = None

    return encoder_model, reranker, bot

def get_context(question, json_file_path, encoder_model, reranker, model_type, n_jump=False,top_k_rerank=5):
    data = load_json(json_file_path)
    if model_type == 'bgem3':
        query_embeddings = encoder_model.encode(question)['dense_vecs']
    elif model_type in ['bce', 'piccolo']:
        query_embeddings = encoder_model.encode(question)
        if isinstance(query_embeddings, np.ndarray):
            query_embeddings = query_embeddings.tolist()

    top_k_embeddings = querypath2node(json_file_path, query_embeddings, top_k=5, model_type=model_type)
    
    adj, title2index = json2adj(json_file_path)
    index_dict = index_by_number(title2index)
    
    title_name_list = []
    dealed_top_k_embeddings = []
    
    for node_title in top_k_embeddings:
        if '_c' in node_title:
            node_title = node_title.split('_c')[0]
        char_title = index_dict[node_title][0][0].split('/')[-1]
        title_name_list.append(char_title)
        dealed_top_k_embeddings.append(node_title)

    if n_jump:
        expanded_embeddings = []
        for node_title in dealed_top_k_embeddings:
            expanded_embeddings.append(node_title)
            descendants = find_node_and_get_descendants(data, node_title)
            if descendants:
                expanded_embeddings.extend(descendants)

        dealed_top_k_embeddings = list(set(expanded_embeddings))  # 去重

        print(f"扩展后的节点列表: {dealed_top_k_embeddings}")

    contexts = get_context_by_node(data, dealed_top_k_embeddings)

    if reranker:
        pairs = [[question, context] for context in contexts.values()]
        scores = reranker.compute_score(pairs)
        
        # 将标题、context和得分组合,并按得分降序排序
        ranked_results = sorted(zip(pairs, scores), key=lambda x: x[1], reverse=True)
        
        # 获取精排后的top1结果
        contexts = [r[0][1] for r in ranked_results[:top_k_rerank]]
    
    return contexts

def file_monitor(directory_path, outdirectory, json_path, model_type, output_json_path, queue):
    last_modified_time = os.path.getmtime(directory_path)
    while True:
        current_modified_time = os.path.getmtime(directory_path)
        if current_modified_time > last_modified_time:
            print("检测到新文件，正在处理...")
            process_documents(directory_path, outdirectory, json_path, model_type, output_json_path)
            last_modified_time = current_modified_time
            queue.put("更新完成")
        time.sleep(5)

def signal_handler(sig, frame):
    print("正在退出程序...")
    try:
        if 'monitor_process' in globals():
            if monitor_process and monitor_process.is_alive():
                monitor_process.terminate()
                monitor_process.join(timeout=5)
                if monitor_process.is_alive():
                    print("强制终止监控进程")
                    monitor_process.kill()
            else:
                print("监控进程已经不存在或已经停止")
        else:
            print("监控进程未创建")
    except Exception as e:
        print(f"清理过程中发生错误: {e}")
    finally:
        sys.exit(0)
# 新增代码: 扩展dealed_top_k_embeddings以包含所有后代节点
def get_all_descendants(node, current_path=''):
    descendants = []
    if 'embeddings' in node:
        for embedding in node['embeddings']:
            descendants.append(embedding)
    
    if 'subheaders' in node:
        for subheader in node['subheaders']:
            # new_path = current_path + subheader['embeddings'][0] + '-' if 'embeddings' in subheader else current_path
            descendants.extend(get_all_descendants(subheader, ''))
    
    return descendants

def find_node_and_get_descendants(data, node_title):
    for item in data:
        if 'embeddings' in item and item['embeddings'][0] == node_title:
            return get_all_descendants(item)
        if 'subheaders' in item:
            result = find_node_and_get_descendants(item['subheaders'], node_title)
            if result:
                return result
    return None


def main(config):
    directory_path = "./LynSDK"
    outdirectory = "./lynSDK_md"
    json_path = "./lynsdk_toc.json"
    model_type = 'bgem3'
    output_json_path = './lynSDK_toc_emb.json'
    
    # 从配置中获取参数
    local_model = config.get('local_model', False)
    first_flag = config.get('first_flag', True)
    n_jump = config.get('n_jump', False)
    top_k_rerank = config.get('top_k_rerank', 5)  # 新增: 从配置中获取top_k_rerank参数
    process_docx = config.get('process_docx', True)
    process_pdf = config.get('process_pdf', True)

    # 初始处理文档
    if first_flag:
        process_documents(directory_path, outdirectory, json_path, model_type, output_json_path, process_docx, process_pdf)

    # 设置模型
    encoder_model, reranker, bot = setup_models(model_type, local_model,n_jump)

    # 创建一个队列用于进程间通信
    queue = Queue()

    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    global monitor_process
    monitor_process = None  # 初始化为None
    try:
        monitor_process = Process(target=file_monitor, args=(directory_path, outdirectory, json_path, model_type, output_json_path, queue))
        monitor_process.start()

            
        while True:
            # 检查队列是否有更新消息
            if not queue.empty():
                message = queue.get()
                print(message)

            # 用户输入问题
            question = input("请输入您的问题（输入'退出'结束）：")
            if question.lower() == '退出':
                break

            # 获取上下文
            contexts = get_context(question, output_json_path, encoder_model, reranker,model_type, n_jump, top_k_rerank)  # 修改: 传入top_k_rerank参数

            # 使用LLM API进行问答
            if local_model:
                messages = [{'role': 'user', 'content': str({'context': contexts, 'query': question})}]
                llm_response = bot.run(messages)
                llm_answer = None
                if llm_response:
                    for response in llm_response:
                        llm_answer = response[0]['content']
            else:
                # llm_answer = get_response(question, contexts)
                llm_answer = get_agent_response(question, contexts)
            # 获取生成器的最后一个输出
            

            print(f'问题: {question}\n答案: {llm_answer}\n')

    except KeyboardInterrupt:
        print("检测到键盘中断，正在退出...")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        print("正在清理资源...")
        try:
            if monitor_process and monitor_process.is_alive():
                monitor_process.terminate()
                monitor_process.join(timeout=5)
                if monitor_process.is_alive():
                    print("强制终止监控进程")
                    monitor_process.kill()
            else:
                print("监控进程已经不存在或已经停止")
        except Exception as e:
            print(f"清理过程中发生错误: {e}")
        print("程序已完全退出")

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RAG系统')
    parser.add_argument('--first_flag', type=lambda x: (str(x).lower() == 'true'), default=False, help='是否进行初始文档处理')
    parser.add_argument('--local_model', type=lambda x: (str(x).lower() == 'true'), default=False, help='是否使用本地模型')
    parser.add_argument('--n_jump', type=lambda x: (str(x).lower() == 'true'), default=True, help='是否启用n_jump')
    parser.add_argument('--top_k_rerank', type=int, default=5, help='重排序后保留的top k结果数')  # 新增: top_k_rerank参数
    parser.add_argument('--process_docx', type=lambda x: (str(x).lower() == 'true'), default=True, help='是否处理docx文件')
    parser.add_argument('--process_pdf', type=lambda x: (str(x).lower() == 'true'), default=True, help='是否处理pdf文件')

    # 解析命令行参数
    args = parser.parse_args()

    # 创建配置字典
    config = {
        'first_flag': args.first_flag,
        'local_model': args.local_model,
        'n_jump': args.n_jump,
        'top_k_rerank': args.top_k_rerank,  # 新增: 将top_k_rerank添加到配置中
        'process_docx': args.process_docx,
        'process_pdf': args.process_pdf
    }
    print(config)
    
    try:
        main(config)
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        try:
            if 'monitor_process' in globals() and monitor_process and monitor_process.is_alive():
                monitor_process.terminate()
                monitor_process.join(timeout=5)
                if monitor_process.is_alive():
                    print("强制终止监控进程")
                    monitor_process.kill()
            else:
                print("监控进程已经不存在或已经停止")
        except Exception as e:
            print(f"清理过程中发生错误: {e}")
        sys.exit(0)