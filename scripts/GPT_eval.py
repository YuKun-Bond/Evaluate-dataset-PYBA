import os
import openai
import re
import json
from tqdm import tqdm

openai.api_key = os.environ.get('OpenAI_SECRET_KEY')

def process_and_save(data, keywords=["總分", "總評分", '最終分數', '最終評分', '綜合評分', '總得分', '總得分', '最終評分']):
    # 以 '\n' 分段
    segments = data.split('\n\n')

    # 搜尋段落中是否有關鍵詞
    for segment in segments:
        if any(keyword in segment for keyword in keywords):
            return segment  # 找到關鍵段落，回傳

    # 如果沒有找到，回傳整段資料
    return data

model_name_number = [150, 300, 600, 1200, 2000]
for number in model_name_number:
    for number2 in range(1, 6, 1):
        model_name = f'PYBA-{number}'
        print(f'PYBA-generated-data/{model_name}-4bit-generated-data-{number2}.json')

        with open(f'PYBA-generated-data/{model_name}-4bit-generated-data-{number2}.json', 'r', encoding='utf-8') as json_file:
            loaded_list = json.load(json_file)

        outputs_list = []
        bar = tqdm(total=10)

        #迴1 - 01
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[0])
        Q = "Q:\n" + statement[1].replace('(有效迴圈的定義為其中的程式運行有必要性，\n無效迴圈範例：\nwhile True:\n  break\n)', '') + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        with open('Scoring-criteria\有效迴圈判斷.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()

        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "請你扮演一個評分專家，依照評分標準評分"},
                {"role": "user", "content": prompt + "\n\n" + Q + "\n" + A}
            ],
            temperature=1,          # 控制創意性，數值範圍是0.0到2.0
            max_tokens=2048,        # 控制回應的長度
            frequency_penalty=0,    # 控制模型避免重複，範圍是-2.0到2.0
            presence_penalty=0      # 控制話題多樣性，範圍是-2.0到2.0
        )

        #紀錄評分文字
        updated_assistant_reply = completion['choices'][0]['message']['content']
        outputs_list.append(process_and_save(updated_assistant_reply))
        bar.update(1)

        #迴2 - 02
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[1])
        Q = "Q:\n" + statement[1].replace('(有效迴圈的定義為其中的程式運行有必要性，\n無效迴圈範例：\nwhile True:\n  break\n)', '') + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "請你扮演一個評分專家，依照評分標準評分"},
                {"role": "user", "content": prompt + "\n\n" + Q + "\n" + A}
            ],
            temperature=1,          # 控制創意性，數值範圍是0.0到2.0
            max_tokens=2048,        # 控制回應的長度
            frequency_penalty=0,    # 控制模型避免重複，範圍是-2.0到2.0
            presence_penalty=0      # 控制話題多樣性，範圍是-2.0到2.0
        )

        #紀錄評分文字
        updated_assistant_reply = completion['choices'][0]['message']['content']
        outputs_list.append(process_and_save(updated_assistant_reply))
        bar.update(1)

        #語1 - 03
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[2])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        with open('Scoring-criteria\語法錯誤判斷.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()

        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "請你扮演一個評分專家，依照評分標準評分"},
                {"role": "user", "content": prompt + "\n\n" + Q + "\n" + A}
            ],
            temperature=1,          # 控制創意性，數值範圍是0.0到2.0
            max_tokens=2048,        # 控制回應的長度
            frequency_penalty=0,    # 控制模型避免重複，範圍是-2.0到2.0
            presence_penalty=0      # 控制話題多樣性，範圍是-2.0到2.0
        )

        #紀錄評分文字
        updated_assistant_reply = completion['choices'][0]['message']['content']
        outputs_list.append(process_and_save(updated_assistant_reply))
        bar.update(1)

        #語2 - 04
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[3])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "請你扮演一個評分專家，依照評分標準評分"},
                {"role": "user", "content": prompt + "\n\n" + Q + "\n" + A}
            ],
            temperature=1,          # 控制創意性，數值範圍是0.0到2.0
            max_tokens=2048,        # 控制回應的長度
            frequency_penalty=0,    # 控制模型避免重複，範圍是-2.0到2.0
            presence_penalty=0      # 控制話題多樣性，範圍是-2.0到2.0
        )

        #紀錄評分文字
        updated_assistant_reply = completion['choices'][0]['message']['content']
        outputs_list.append(process_and_save(updated_assistant_reply))
        bar.update(1)

        #邏1 - 05
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[4])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        with open('Scoring-criteria\邏輯錯誤判斷.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()

        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "請你扮演一個評分專家，依照評分標準評分"},
                {"role": "user", "content": prompt + "\n\n" + Q + "\n" + A}
            ],
            temperature=1,          # 控制創意性，數值範圍是0.0到2.0
            max_tokens=2048,        # 控制回應的長度
            frequency_penalty=0,    # 控制模型避免重複，範圍是-2.0到2.0
            presence_penalty=0      # 控制話題多樣性，範圍是-2.0到2.0
        )

        #紀錄評分文字
        updated_assistant_reply = completion['choices'][0]['message']['content']
        outputs_list.append(process_and_save(updated_assistant_reply))
        bar.update(1)

        #邏2 - 06
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[5])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "請你扮演一個評分專家，依照評分標準評分"},
                {"role": "user", "content": prompt + "\n\n" + Q + "\n" + A}
            ],
            temperature=1,          # 控制創意性，數值範圍是0.0到2.0
            max_tokens=2048,        # 控制回應的長度
            frequency_penalty=0,    # 控制模型避免重複，範圍是-2.0到2.0
            presence_penalty=0      # 控制話題多樣性，範圍是-2.0到2.0
        )

        #紀錄評分文字
        updated_assistant_reply = completion['choices'][0]['message']['content']
        outputs_list.append(process_and_save(updated_assistant_reply))
        bar.update(1)

        #描1 - 07
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[6])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        with open('Scoring-criteria\清晰的描述程式碼邏輯結構.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()

        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "請你扮演一個評分專家，依照評分標準評分"},
                {"role": "user", "content": prompt + "\n\n" + Q + "\n" + A}
            ],
            temperature=1,          # 控制創意性，數值範圍是0.0到2.0
            max_tokens=2048,        # 控制回應的長度
            frequency_penalty=0,    # 控制模型避免重複，範圍是-2.0到2.0
            presence_penalty=0      # 控制話題多樣性，範圍是-2.0到2.0
        )

        #紀錄評分文字
        updated_assistant_reply = completion['choices'][0]['message']['content']
        outputs_list.append(process_and_save(updated_assistant_reply))
        bar.update(1)

        #描2 - 08
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[7])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "請你扮演一個評分專家，依照評分標準評分"},
                {"role": "user", "content": prompt + "\n\n" + Q + "\n" + A}
            ],
            temperature=1,          # 控制創意性，數值範圍是0.0到2.0
            max_tokens=2048,        # 控制回應的長度
            frequency_penalty=0,    # 控制模型避免重複，範圍是-2.0到2.0
            presence_penalty=0      # 控制話題多樣性，範圍是-2.0到2.0
        )

        #紀錄評分文字
        updated_assistant_reply = completion['choices'][0]['message']['content']
        outputs_list.append(process_and_save(updated_assistant_reply))
        bar.update(1)

        #完1 - 09
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[8])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        with open('Scoring-criteria\完成指定功能程式碼.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()

        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "請你扮演一個評分專家，依照評分標準評分"},
                {"role": "user", "content": prompt + "\n\n" + Q + "\n" + A}
            ],
            temperature=1,          # 控制創意性，數值範圍是0.0到2.0
            max_tokens=2048,        # 控制回應的長度
            frequency_penalty=0,    # 控制模型避免重複，範圍是-2.0到2.0
            presence_penalty=0      # 控制話題多樣性，範圍是-2.0到2.0
        )

        #紀錄評分文字
        updated_assistant_reply = completion['choices'][0]['message']['content']
        outputs_list.append(process_and_save(updated_assistant_reply))
        bar.update(1)

        #完2 - 10
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[9])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        with open('Scoring-criteria\完成指定功能程式碼.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()

        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "請你扮演一個評分專家，依照評分標準評分，在回應的最後以總分結尾"},
                {"role": "user", "content": prompt + "\n\n" + Q + "\n" + A}
            ],
            temperature=1,          # 控制創意性，數值範圍是0.0到2.0
            max_tokens=2048,        # 控制回應的長度
            frequency_penalty=0,    # 控制模型避免重複，範圍是-2.0到2.0
            presence_penalty=0      # 控制話題多樣性，範圍是-2.0到2.0
        )

        #紀錄評分文字
        updated_assistant_reply = completion['choices'][0]['message']['content']
        outputs_list.append(process_and_save(updated_assistant_reply))
        bar.update(1)

        with open(f'Scoring-result/{model_name}-{number2}.json', 'w', encoding='utf-8') as json_file:
            json.dump(outputs_list, json_file, ensure_ascii=False, indent=2)