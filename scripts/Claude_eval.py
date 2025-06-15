import os
import openai
import re
import json
from tqdm import tqdm

# openai.api_key = os.environ.get('OpenAI_SECRET_KEY')

def process_and_save(data, keywords=["總分", "總評分", '最終分數', '最終評分', '綜合評分', '總得分', '總得分', '最終評分']):
    # 以 '\n' 分段
    segments = data.split('\n\n')

    # 搜尋段落中是否有關鍵詞
    for segment in segments:
        if any(keyword in segment for keyword in keywords):
            return segment  # 找到關鍵段落，回傳

    # 如果沒有找到，回傳整段資料
    return data

def get_score_by_model(model_type, prompt):
    if model_type == "gpt":
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "請你扮演一個評分專家，依照評分標準評分"},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=2048,
            frequency_penalty=0,
            presence_penalty=0
        )
        return completion['choices'][0]['message']['content']

    elif model_type == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["Google_AI_Studio_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        return response.text

    elif model_type == "claude":
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
        client = Anthropic(api_key=os.environ["Claude-API-KEY"])
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2048,
            temperature=1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text  # 回傳文字內容

    else:
        raise ValueError("不支援的模型類型")

model_names = ['meta-llama_Meta-Llama-3-8B-Instruct', 'meta-llama_Meta-Llama-3-8B', 'google_gemma-2-9b', 'mistralai_Mistral-7B-Instruct-v0.3', 'yentinglin_Llama-3-Taiwan-8B-Instruct', 'yentinglin_Taiwan-LLM-7B-v2.0.1-chat', 'microsoft_Phi-3-mini-4k-instruct']

for model_name in model_names:
    for number2 in range(1, 2, 1):
        print(f'模型初步評估資料/{model_name}.json')

        with open(f'模型初步評估資料/{model_name}.json', 'r', encoding='utf-8') as json_file:
            loaded_list = json.load(json_file)

        if len(loaded_list) == 10:
            print("正常：有十個項目")
        else:
            print("錯誤：只有" + len(loaded_list) +"個項目")

        outputs_list = []
        all_outputs_list = []
        bar = tqdm(total=10)

        #迴1 - 01
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[0])
        Q = "Q:\n" + statement[1].replace('(有效迴圈的定義為其中的程式運行有必要性，\n無效迴圈範例：\nwhile True:\n  break\n)', '') + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        with open('Scoring-criteria\有效迴圈判斷.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()

        #紀錄評分文字
        updated_assistant_reply = get_score_by_model("claude", prompt + "\n\n" + Q + "\n" + A)
        outputs_list.append(process_and_save(updated_assistant_reply))
        all_outputs_list.append((updated_assistant_reply))
        bar.update(1)

        #迴2 - 02
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[1])
        Q = "Q:\n" + statement[1].replace('(有效迴圈的定義為其中的程式運行有必要性，\n無效迴圈範例：\nwhile True:\n  break\n)', '') + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        #紀錄評分文字
        updated_assistant_reply = get_score_by_model("claude", prompt + "\n\n" + Q + "\n" + A)
        outputs_list.append(process_and_save(updated_assistant_reply))
        all_outputs_list.append((updated_assistant_reply))
        bar.update(1)

        #語1 - 03
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[2])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        with open('Scoring-criteria\語法錯誤判斷.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()

        #紀錄評分文字
        updated_assistant_reply = get_score_by_model("claude", prompt + "\n\n" + Q + "\n" + A)
        outputs_list.append(process_and_save(updated_assistant_reply))
        all_outputs_list.append((updated_assistant_reply))
        bar.update(1)

        #語2 - 04
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[3])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        #紀錄評分文字
        updated_assistant_reply = get_score_by_model("claude", prompt + "\n\n" + Q + "\n" + A)
        outputs_list.append(process_and_save(updated_assistant_reply))
        all_outputs_list.append((updated_assistant_reply))
        bar.update(1)

        #邏1 - 05
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[4])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        with open('Scoring-criteria\邏輯錯誤判斷.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()

        #紀錄評分文字
        updated_assistant_reply = get_score_by_model("claude", prompt + "\n\n" + Q + "\n" + A)
        outputs_list.append(process_and_save(updated_assistant_reply))
        all_outputs_list.append((updated_assistant_reply))
        bar.update(1)

        #邏2 - 06
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[5])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        #紀錄評分文字
        updated_assistant_reply = get_score_by_model("claude", prompt + "\n\n" + Q + "\n" + A)
        outputs_list.append(process_and_save(updated_assistant_reply))
        all_outputs_list.append((updated_assistant_reply))
        bar.update(1)

        #描1 - 07
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[6])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        with open('Scoring-criteria\清晰的描述程式碼邏輯結構.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()

        #紀錄評分文字
        updated_assistant_reply = get_score_by_model("claude", prompt + "\n\n" + Q + "\n" + A)
        outputs_list.append(process_and_save(updated_assistant_reply))
        all_outputs_list.append((updated_assistant_reply))
        bar.update(1)

        #描2 - 08
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[7])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        #紀錄評分文字
        updated_assistant_reply = get_score_by_model("claude", prompt + "\n\n" + Q + "\n" + A)
        outputs_list.append(process_and_save(updated_assistant_reply))
        all_outputs_list.append((updated_assistant_reply))
        bar.update(1)

        #完1 - 09
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[8])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        with open('Scoring-criteria\完成指定功能程式碼.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()

        #紀錄評分文字
        updated_assistant_reply = get_score_by_model("claude", prompt + "\n\n" + Q + "\n" + A)
        outputs_list.append(process_and_save(updated_assistant_reply))
        all_outputs_list.append((updated_assistant_reply))
        bar.update(1)

        #完2 - 10
        statement = re.split(r'### Instruction:\n|### Input:\n|### Response:\n', loaded_list[9])
        Q = "Q:\n" + statement[1] + statement[2]
        A = "A:\n" + statement[3].replace('<|end_of_text|>', '')

        with open('Scoring-criteria\完成指定功能程式碼.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()

        #紀錄評分文字
        updated_assistant_reply = get_score_by_model("claude", prompt + "\n\n" + Q + "\n" + A)
        outputs_list.append(process_and_save(updated_assistant_reply))
        all_outputs_list.append((updated_assistant_reply))
        bar.update(1)

        with open(f'First-Scoring-result-Claude/{model_name}.json', 'w', encoding='utf-8') as json_file:
            json.dump(outputs_list, json_file, ensure_ascii=False, indent=2)

        with open(f'First-Scoring-result-Claude-ALL/{model_name}-ALL.json', 'w', encoding='utf-8') as json_file:
            json.dump(all_outputs_list, json_file, ensure_ascii=False, indent=2)