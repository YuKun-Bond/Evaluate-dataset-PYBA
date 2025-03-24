import os
import openai
import re
import json
from tqdm import tqdm

with open(f'./PYBA-generated-data/PYBA-2000-4bit-generated-data-5.json', 'r', encoding='utf-8') as json_file:
    loaded_list = json.load(json_file)

for loaded in loaded_list:
    print(loaded)
    input()