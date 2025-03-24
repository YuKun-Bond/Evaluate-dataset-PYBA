import json

# 讀取原始JSONL檔案
with open('training.jsonl', 'r', encoding='utf-8') as infile:
    data = []
    for line in infile:
        data.append(json.loads(line.strip()))

transformed_data = []

for entry in data:
    transformed_entry = {
        "instruction": entry["instruction"],
        "input": entry["input"],
        "output": entry["output"]
    }
    transformed_data.append(transformed_entry)

# 將轉換後的資料寫入新JSONL檔案，每筆資料單獨一行
with open('training.json', 'w', encoding='utf-8') as outfile:
    json.dump(transformed_data, outfile, ensure_ascii=False, indent=2)

print("資料轉換完成並儲存至 'transformed-fine-tuning-data.jsonl'")
