import os 
import tqdm 
import requests
import pandas as pd

PREFIX = "http://arxiv.org/abs/"
ADDRESS = "http://export.arxiv.org"

with open("withdrawn.txt", "r") as f:
    lines = f.readlines()
    withdrawn_ids = list(map(int, lines))

df = pd.read_csv('../../data/arxiv/arxiv_dataset_all_info.csv')

# TODO fix to save with arxiv ids
for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):

    filename = f"PDF/{i}.pdf"
    if os.path.exists(filename):
        print(f"{i}: {filename} exists, skip")
        continue

    idx = row['id']
    if i in withdrawn_ids:
        print(f"{i}: {idx} is withdrawn, skip")
        continue

    address = f"{ADDRESS}/pdf/{idx[len(PREFIX):]}"
    response = requests.get(address)   

    if response.status_code != 200:
        print(address)
        print(f"{i} failed")
        break 
    else:
        print(f"{i} success")
    
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"{i}: {filename} saved")
          
