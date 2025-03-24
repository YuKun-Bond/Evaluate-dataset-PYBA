import os
import json

dataset_list = ['有效迴圈判斷.json', '語法錯誤判斷.json', '邏輯錯誤判斷.json', '程式碼優化.json', '清晰的描述程式碼邏輯結構.json', '完成指定功能程式碼.json', 'dataset_180.json', 'dataset_360.json', 'dataset_720.json', 'dataset_1440.json', 'dataset_2400.json']
for i in range(len(dataset_list)):
    dataset_file_path = r'./PYPD-dataset/' + dataset_list[i]

    # 打開現有的JSON文件並讀取內容
    with open(dataset_file_path, 'r', encoding='utf-8') as json_file:
        existing_data = json.load(json_file)
    
    print(len(existing_data))