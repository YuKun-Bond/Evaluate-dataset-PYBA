import os
import google.generativeai as genai
import re
import json
from tqdm import tqdm

api_key = os.environ.get('Google_AI_Studio_API_KEY')
genai.configure(api_key = api_key)

def remove_last_code_block_ending(text):
    # 使用正則表達式找到最後一個 "```" 的位置
    pattern = re.compile(r'```(?!.*```)', re.DOTALL)
    match = pattern.search(text)
    
    if match:
        # 找到最後一個 "```" 的位置，並移除它及其後所有內容
        return text[:match.end()]
    return text

def remove_last_code_block_ending2(text):
    # 使用正則表達式找到最後一個 "}" 的位置
    pattern = re.compile(r'}(?!.*})', re.DOTALL)
    match = pattern.search(text)
    
    if match:
        # 找到最後一個 "}" 的位置，並移除它及其後所有內容
        return text[:match.end()]
    return text

prompt_list = ['描述範例Prompt.txt', '完成範例Prompt.txt']
dataset_list = ['清晰的描述程式碼邏輯結構.json', '完成指定功能程式碼.json']

for i in range(len(prompt_list)):

    prompt_file_path = r'./範例產生Prompt/' + prompt_list[i]
    dataset_file_path = r'./PYPD-dataset/' + dataset_list[i]

    with open(prompt_file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    
    bar = tqdm(total=400)
    print()

    while True:
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(f'{prompt}' + "\n請給我json樣式的資料，裡面包含instruction、input、output的key與其對應的value")

            updated_assistant_reply = response.text

            # 使用正則表達式提取包在```json與```之間的內容
            json_pattern = re.search(r'```json\n(.*?)(?=$)', updated_assistant_reply, re.DOTALL)
            if json_pattern:
                json_content = remove_last_code_block_ending(json_pattern.group(1).strip().replace('[', '').replace('\n\n', '\n'))

            json_pattern = re.search(r'{\n(.*?)(?=$)', updated_assistant_reply, re.DOTALL)
            if json_pattern:
                json_content = '{' + remove_last_code_block_ending2(json_pattern.group(1).strip().replace('[', '').replace('\n\n', '\n')) + '}'
            else:
                json_content = updated_assistant_reply.replace('[', '').replace('\n\n', '\n')

            json_content = json.loads(json_content)
            # print('json_content', json_content)

            # 打開現有的JSON文件並讀取內容
            with open(dataset_file_path, 'r', encoding='utf-8') as json_file:
                existing_data = json.load(json_file)
            # print('existing_data', existing_data)

            # 將新的內容追加到現有的JSON數據中
            existing_data.append(json_content)

            # 將更新後的數據寫回文件
            with open(dataset_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

            dataset_name = dataset_list[i].replace('.json', '')
            print(f'{dataset_name}資料集 - 資料筆數', len(existing_data))

            bar.n = len(existing_data)  # 直接設定進度
            bar.refresh()
            print()
            
            if len(existing_data) >= 400:
                break

        except Exception as e:
            print('updated_assistant_reply', updated_assistant_reply)
            print('json_content', json_content)
            print(e)