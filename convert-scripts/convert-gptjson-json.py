import json

# 讀取原始 JSON 檔案
with open('validation-dataset.json', 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# 初始化空列表來存放轉換後的資料
transformed_data = []

# 迭代處理每一個資料項目
for entry in data:
    system_message = entry["messages"][0]["content"]
    user_message = entry["messages"][1]["content"]
    assistant_message = entry["messages"][2]["content"]

    # 構造轉換後的項目
    transformed_entry = {
        "instruction": system_message,
        "input": user_message,
        "output": assistant_message
    }

    # 將轉換後的項目加入列表
    transformed_data.append(transformed_entry)

# 將轉換後的資料寫入新的 JSON 檔案
with open('validation-dataset-new.json', 'w', encoding='utf-8') as outfile:
    json.dump(transformed_data, outfile, ensure_ascii=False, indent=2)

print("資料轉換完成，並已存為 'validation-dataset-new.json'。")
