import json

# 讀取原始JSON檔案
with open('training-dataset.json', 'r', encoding='utf-8') as infile:
    data = json.load(infile)

transformed_data = []

for entry in data:
    transformed_entry = {
        "messages": [
            { "role": "system", "content": entry["instruction"] },
            { "role": "user", "content": entry["input"] },
            { "role": "assistant", "content": entry["output"] }
        ]
    }
    transformed_data.append(transformed_entry)

# 將轉換後的資料寫入新JSON檔案
with open('training-dataset-gpt.json', 'w', encoding='utf-8') as outfile:
    json.dump(transformed_data, outfile, ensure_ascii=False, indent=2)

print("資料轉換完成並儲存至 'training-dataset-gpt.json'")
