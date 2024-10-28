import json
from collections import defaultdict
from datetime import datetime, timedelta
from timeit import main

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def parse_date(date_str,date):
    if date_str == "去年":
        # 假设"去年"是指前一年的最后一天
        current_year = date.split('-')[0]
        return datetime(int(current_year) - 1, 12, 31)
    try:
        if date_str:
            return datetime.strptime(date_str, "%Y-%m-%d")
        else:
            return datetime(1900, 1, 1)
    except ValueError:
        # 如果无法解析日期，返回一个很早的日期
        return datetime(1900, 1, 1)

def sort_and_organize_data(data):
    organized_data = defaultdict(list)
    
    for date, records in data['daily_records'].items():
        for record in records:
            for person, info_list in record.items():
                if person != '原文':        
                    for info in info_list:
                        info_date = parse_date(info['date'], date)
                        organized_data[info_date].append({
                            'person': person,
                            'date': info['date'],  # 保留原始日期字符串
                            'location': info['location'],
                            'events': info['events'],
                            'relationships': info['relationships']
                    })  
    
    # 按日期排序
    sorted_data = dict(sorted(organized_data.items()))
    
    # 将datetime键转换回字符串
    result = {k.strftime("%Y-%m-%d") if isinstance(k, datetime) else str(k): v for k, v in sorted_data.items()}
    
    # 添加overall_relationships到结果中
    result['overall_relationships'] = data['overall_relationships']
    
    return result

def save_sorted_data(sorted_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(sorted_data, file, ensure_ascii=False, indent=2)

def date_translate():
    input_file = 'chat_analysis_results.json'
    output_file = 'sorted_chat_analysis_results.json'
    
    data = load_json_data(input_file)
    sorted_data = sort_and_organize_data(data)
    save_sorted_data(sorted_data, output_file)
    
    print(f"数据已按日期排序并保存到 {output_file}")

if __name__ == "__main__":
    date_translate()