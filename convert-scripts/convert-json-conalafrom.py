import json

# 讀取原始JSON檔案
with open('validation-dataset.json', 'r', encoding='utf-8') as infile:
    data = json.load(infile)

transformed_data = []

for entry in data:
    transformed_entry = {"intent": entry["instruction"],"rewritten_intent": entry["instruction"] + "\n" + entry["input"],"snippet": entry["output"],"question_id":0}
    transformed_data.append(transformed_entry)

# 將轉換後的資料寫入新JSON檔案，每筆資料單獨一行
with open('validation-dataset-conalafrom.jsonl', 'w', encoding='utf-8') as outfile:
    for item in transformed_data:
        json.dump(item, outfile, ensure_ascii=False)
        outfile.write('\n')

print("資料轉換完成並儲存")
