import os
import sys 
import json
import tqdm
import pandas as pd
import logging 

from alex_utils import clean_title, get_openalex_id_from_arxiv, get_openalex_id_from_title

DATA_DIR = "../../data"
RESULTS = "../../results"

df = pd.read_csv(f"{DATA_DIR}/arxiv/arxiv_dataset_all_info.csv")


di_scores = pd.read_feather(f"{DATA_DIR}/openalex_di/OpenAlexID_Year_DindexTenYears.feather")

arxiv2alex = dict()
if os.path.exists("arxiv2alex.json"):
    with open("arxiv2alex.json", "r") as f:
        arxiv2alex = json.load(f)

result = {}
        
for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    arxiv_id = row["id"]
    title =  row["title"]

    if arxiv_id in arxiv2alex:
        alex_ids = arxiv2alex[arxiv_id]
    else:
        alex_ids = get_openalex_id_from_arxiv(arxiv_id)
        if alex_ids is None:
            alex_ids = get_openalex_id_from_title(title) 
    
        alex_ids = list(map(lambda x: x.split('/')[-1], alex_ids))
        arxiv2alex[arxiv_id] = alex_ids
        with open("arxiv2alex.json", "w") as f:
            json.dump(arxiv2alex, f)

    di_values = list(di_scores.loc[di_scores["alex_id"].isin(alex_ids), "score"])
    if di_values:
        result[arxiv_id] = max(di_values)
    
    
result = [{"id": key, "novelty_score": value} for key, value in result.items()] 
pd.DataFrame(result).to_csv(f"{RESULTS}/di_values_10years_result.csv")


