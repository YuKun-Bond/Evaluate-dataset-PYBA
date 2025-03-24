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
        # 找到最後一個 "```" 的位置，並移除它
        return text[:match.start()]  # 保留最後一個 "```" 之前的內容
    return text

prompt = '''有效迴圈判斷範例

提供特定語法學習後之應用能力，例如不濫用迴圈，確保LLM可偵測到正確的迴圈使用數量。

以下是有效迴圈與無效迴圈的定義

有效迴圈是指其運行有必要性，並且對程式的功能或行為產生實際影響的迴圈。這些迴圈通常執行某種重複的操作或
處理數據，並且其結構和條件設計合理，避免無限迴圈或無用的運行。

無效迴圈是指其運行沒有必要性，對程式的功能或行為沒有實際影響的迴圈。這些迴圈通常包含無效的結構或條件，
例如永遠不會執行的迴圈或立即中斷的迴圈。

有效迴圈:
# 這是一個有效的 for 迴圈，因為它對每個元素進行處理
for i in range(10):
    print(i)

# 這是一個有效的 while 迴圈，因為它根據條件執行操作
x = 0
while x < 5:
    print(x)
    x += 1

無效迴圈:
# 這是一個無效的 while 迴圈，因為它立即中斷
while True:
    break

# 這是一個無效的 while 迴圈，因為它永遠不會執行
while False:
    print("This will never print")

這是我的評分標準 請盡量完成
準確性 正確描述程式碼的有效迴圈數量
詳細性 詳細描述每個迴圈的功能與判斷為無效迴圈的理由
正向建議 提供無效迴圈刪除建議、巢狀迴圈使用與迴圈類型使用建議

範例:
{
  "instruction": "我的程式碼內有幾個有效迴圈？",
  "input": "count = 0\nfor i in range(10):\n  if i > 5:\n    count += i\n\nwhile False:\n  print('This will never print')\n\nwhile True:\n  break",
  "output": "這段程式碼內有一個有效的迴圈和兩個無效的迴圈。\n\n1. **有效的 for 迴圈**：這個 `for` 迴圈有效，因為它遍歷範圍 0 到 9 並在 `i` 大於 5 時累加 `count`。這個迴圈的功能是計算大於 5 的數字之和，對程式有實際意義。\n\n2. **無效的 while 迴圈 (第一個)**：這個 `while` 迴圈無效，因為條件是 `False`，因此迴圈的主體永遠不會執行。這樣的結構不會對程式的行為產生任何影響。\n\n3. **無效的 while 迴圈 (第二個)**：這個 `while` 迴圈同樣無效，因為它在第一次迭代時立即中斷 (`break`)。這樣的設計並沒有實際的功能，等同於無用的程式碼。\n\n**正向建議**：\n\n- **刪除無效迴圈**：應該刪除兩個無效的 `while` 迴圈，因為它們不會對程式產生任何實際作用。\n- **巢狀迴圈使用建議**：如果未來需要根據更複雜的條件進行多層次的數據處理，可以考慮使用巢狀迴圈。例如，可以在有效的 `for` 迴圈內嵌套另一個 `for` 迴圈，來對某些條件進行進一步的運算。\n- **迴圈類型使用建議**：在需要進行明確次數的迴圈操作時，使用 `for` 迴圈較為合適；而當需要在未知次數的情況下重複操作時，`while` 迴圈則更為有效，但要注意避免無限迴圈或立即中斷的情況。"
}

請你扮演一個Python專業出題者
給我五個滿足上面格式與標準的範例
每一個範例分別用
```json
......
```
包裹起來
'''

completion = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "請產生符合指定要求的範例，不要有對話或解釋，只回應json內容就好了，json內容文字的部分請使用繁體中文"},
        {"role": "user", "content": prompt}
    ],
    temperature=1,          # 控制創意性，數值範圍是0.0到2.0
    max_tokens=4095,        # 控制回應的長度
    frequency_penalty=0,    # 控制模型避免重複，範圍是-2.0到2.0
    presence_penalty=0      # 控制話題多樣性，範圍是-2.0到2.0
)

updated_assistant_reply = completion['choices'][0]['message']['content']
print(updated_assistant_reply)


updated_assistant_replys = list(map(remove_last_code_block_ending, updated_assistant_reply.split('```json')))

for text in updated_assistant_replys:
    print(json.loads(text))
    print()