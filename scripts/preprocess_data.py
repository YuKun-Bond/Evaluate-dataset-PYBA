import json

def preprocess_data(entry):
    # 將 output 字段轉換為 JSON 字符串格式
    if isinstance(entry.get('output'), dict):
        entry['output'] = json.dumps(entry['output'], ensure_ascii=False)
    return entry

def load_and_process_dataset(file_path):
    # 加載資料集
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 遍歷每個數據項進行預處理
    processed_dataset = [preprocess_data(entry) for entry in dataset]

    # 將修改後的資料集寫回原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_dataset, f, ensure_ascii=False, indent=4)

    print(f"已成功處理並覆蓋保存 {file_path}")

# 資料集大小
sizes = [180, 360, 720, 1440, 2400]
for size in sizes:
  # 使用範例，將 'dataset.json' 替換為你的資料集文件名
  file_path = f"PYPD-dataset\\dataset_{size}.json"
  load_and_process_dataset(file_path)
