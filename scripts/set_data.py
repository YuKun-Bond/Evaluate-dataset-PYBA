import json

# 定義資料樣本
dataset = [
    {
        "description": "我的程式碼內有幾個有效迴圈",
        "code": "for i in range(100):\n  for j in range(20):\n    print('*', end='')\n  print()\n\nfor i in range(50):\n  pass",
        "result": "這段程式碼內有兩個for迴圈被有效使用，一個被無效使用"
    },
    {
        "description": "我的程式碼內有幾個有效迴圈",
        "code": "while True:\n  break\n\nfor i in range(10):\n  for j in range(5):\n    print(i * j)",
        "result": "這段程式碼內有一個while迴圈被無效使用，兩個for迴圈被有效使用"
    },
    {
        "description": "我的程式碼內有幾個有效迴圈",
        "code": "for i in range(10):\n  print(i)\n\nwhile False:\n  print('This will never print')",
        "result": "這段程式碼內有一個for迴圈被有效使用，一個while迴圈被無效使用"
    },
    {
        "description": "我的程式碼內有幾個有效迴圈",
        "code": "for i in range(3):\n  for j in range(4):\n    for k in range(2):\n      print(i, j, k)",
        "result": "這段程式碼內有三個for迴圈被有效使用"
    },
    {
        "description": "我的程式碼內有幾個有效迴圈",
        "code": "while condition:\n  do_something()\n\nfor i in range(5):\n  continue",
        "result": "這段程式碼內有一個while迴圈被有效使用，一個for迴圈被無效使用"
    },
    {
        "description": "我的程式碼內有幾個有效迴圈",
        "code": "for i in range(1, 10, 2):\n  for j in range(2, 20, 2):\n    pass\n\nwhile 1 < 0:\n  pass",
        "result": "這段程式碼內有兩個for迴圈被無效使用，一個while迴圈被無效使用"
    },
    {
        "description": "我的程式碼內有幾個有效迴圈",
        "code": "for i in range(10):\n  while i < 5:\n    print(i)\n    i += 1",
        "result": "這段程式碼內有一個for迴圈被有效使用，一個while迴圈被有效使用"
    },
    {
        "description": "我的程式碼內有幾個有效迴圈",
        "code": "while True:\n  for i in range(3):\n    print('Hello')\n  break",
        "result": "這段程式碼內有一個while迴圈被有效使用，一個for迴圈被有效使用"
    },
    {
        "description": "我的程式碼內有幾個有效迴圈",
        "code": "for x in [1, 2, 3]:\n  for y in [4, 5, 6]:\n    for z in [7, 8, 9]:\n      print(x, y, z)",
        "result": "這段程式碼內有三個for迴圈被有效使用"
    },
    {
        "description": "我的程式碼內有幾個有效迴圈",
        "code": "while x > 0:\n  x -= 1\n\nfor i in range(2):\n  while y < 5:\n    y += 1",
        "result": "這段程式碼內有一個while迴圈被有效使用，一個for迴圈被有效使用，一個while迴圈被有效使用"
    }
]

# 將數據集寫入JSON文件
output_file = 'dataset.json'

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"Dataset saved to {output_file}")