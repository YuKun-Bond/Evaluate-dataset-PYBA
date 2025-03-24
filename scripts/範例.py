# description: 請完成一個可以輸出99乘法表的Python程式碼。

def multiplication_table():
    for i in range(1, 10):
        for j in range(1, 10):
            print(f'{i} * {j} = {i * j}', end='\t')
        print()

# result: 這段程式碼定義了一個函式 multiplication_table()，用於輸出99乘法表。

# 具體來說，函式內包含兩個嵌套的for迴圈。外部迴圈變量 i 從1到9，內部迴圈變量 j 也從1到9。在內部迴圈中，
# 計算 i 和 j 的乘積並使用 print 函式格式化輸出結果，並且用 '\t' 來控制輸出的間隔。外部迴圈每執行一次後，使用 print() 函式換行。
# 最終，這段程式碼會輸出1到9的乘法表。