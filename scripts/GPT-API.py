import os
import openai
import re
import json
from tqdm import tqdm

openai.api_key = os.environ.get('OpenAI_SECRET_KEY')

def remove_last_code_block_ending(text):
    # 使用正則表達式找到最後一個 "```" 的位置
    pattern = re.compile(r'```(?!.*```)', re.DOTALL)
    match = pattern.search(text)
    
    if match:
        # 找到最後一個 "```" 並移除它
        last_code_block_end = match.start()
        return text[:last_code_block_end] + text[last_code_block_end + 3:]
    return text

prompt_list = ['邏輯錯誤判斷', '清晰的描述程式碼邏輯結構', '完成指定功能程式碼']

for i in range(len(prompt_list)):

    prompt_file_path = r'./範例產生Prompt/' + prompt_list[i] + '.txt'
    dataset_file_path = r'./PYPD-dataset/' + prompt_list[i] + '.json'

    with open(prompt_file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()

    bar = tqdm(total=400)

    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "請產生符合指定要求的範例，不要有對話或解釋，只回應json內容就好了，json內容文字的部分請使用繁體中文"},
                    {"role": "user", "content": prompt}
                ],
                temperature=1,          # 控制創意性，數值範圍是0.0到2.0
                max_tokens=1024,        # 控制回應的長度
                frequency_penalty=0,    # 控制模型避免重複，範圍是-2.0到2.0
                presence_penalty=0      # 控制話題多樣性，範圍是-2.0到2.0
            )

            updated_assistant_reply = completion['choices'][0]['message']['content']

            # 使用正則表達式提取包在```json與```之間的內容
            json_pattern = re.search(r'```json\n(.*?)(?=$)', updated_assistant_reply, re.DOTALL)
            if json_pattern:
                json_content = remove_last_code_block_ending(json_pattern.group(1).strip().replace('[', '').replace('\n\n', '\n'))
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

            dataset_name = prompt_list[i]
            print(f'{dataset_name}資料集 - 資料筆數', len(existing_data))

            bar.n = len(existing_data)  # 直接設定進度
            bar.refresh()
            
            if len(existing_data) >= 400:
                break
            

        except Exception as e:
            print('updated_assistant_reply', updated_assistant_reply)
            print('json_content', json_content)
            print(e)