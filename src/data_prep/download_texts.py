import os 
import tqdm 
import requests

from arxiv2text import arxiv_to_text
import pandas as pd

PREFIX = "http://arxiv.org/abs/"
ADDRESS = "http://export.arxiv.org"

with open("withdrawn.txt", "r") as f:
    lines = f.readlines()
    withdrawn_ids = list(map(lines, int))

df = pd.read_csv('arxiv_dataset_all_info.csv')

for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):

    filename = f"TXT/{i}.txt"
    
    if os.path.exists(filename):
        print(f"{i}: {filename} exists, skip")
        continue

    if i in withdrawn_ids:
        print(f"{i}: {idx} is withdrawn, skip")
        continue

    idx = row['id']
    address = f"{ADDRESS}/pdf/{idx[len(PREFIX):]}"
    text = arxiv_to_text(address)
    
    with open(filename, "w") as f:
        print(text, file=f)
    print(f"{i}: {filename} saved")
          
