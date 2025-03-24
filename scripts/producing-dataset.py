import json
import random

# 讀取6個JSON文件的路徑
files = [
    "PYPD-dataset\\完成指定功能程式碼.json",
    "PYPD-dataset\\有效迴圈判斷.json",
    "PYPD-dataset\\清晰的描述程式碼邏輯結構.json",
    "PYPD-dataset\\語法錯誤判斷.json",
    "PYPD-dataset\\邏輯錯誤判斷.json"
]


# "PYPD-dataset\\完成指定功能程式碼.json",
# "PYPD-dataset\\有效迴圈判斷.json",
# "PYPD-dataset\\清晰的描述程式碼邏輯結構.json",
# "PYPD-dataset\\程式碼優化.json",
# "PYPD-dataset\\語法錯誤判斷.json",
# "PYPD-dataset\\邏輯錯誤判斷.json"


# 定義資料集大小
sizes = [150, 300, 600, 1200, 2000]

for size in sizes:
    subset_data = []

    # 從每個文件中提取所需數量的1/6資料
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 計算需要提取的數量
            extract_count = size // 6
            subset_data.extend(data[:extract_count])

    # 隨機打亂順序
    random.shuffle(subset_data)

    # 保存為新文件
    output_file = f"PYPD-dataset\\dataset_{size}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, ensure_ascii=False, indent=4)

print("資料集已成功創建並保存。")
