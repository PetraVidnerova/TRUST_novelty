import os
import sys 
import json
import pickle
import tqdm
import pandas as pd
import logging 

from alex_utils import clean_title, get_openalex_id_from_arxiv, get_openalex_id_from_title
from alex_utils import get_openalex_id_from_pubmed
from utils.config import load_settings
from utils.utils import read_df

DATA_DIR = "../../data"
RESULTS = "../../results"

cfg = load_settings()
ID = cfg.id_column_name

df = read_df(cfg.input_file)
di_scores = pd.read_feather(f"{DATA_DIR}/openalex_di/OpenAlexID_Year_DindexTenYears.feather")

 
id2alex = dict()
if cfg.output_prefix == "arxiv_dataset":
    id2alex_filename = "arxiv2alex.json"
else:
    id2alex_filename = "pmid2alex.json"

    
if os.path.exists(id2alex_filename):
    with open(id2alex_filename, "r") as f:
        id2alex = json.load(f)
        
result = dict()
if os.path.exists("tmp_result.pickle"):
    with open("tmp_result.pickle", "rb") as f:
        result = pickle.load(f)
        print("result loaded")

print(result)

for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    orig_id = row[ID]
    title =  row["title"]

    if orig_id in result:
        print("computed already")
        continue
    print(orig_id)
    
    if orig_id in id2alex:
        alex_ids = id2alex[orig_id]
    else:
        if cfg.output_prefix == "arxiv_dataset": 
            alex_ids = get_openalex_id_from_arxiv(arxiv_id)
        elif cfg.output_prefix == "novelpy_data":
            alex_id = get_openalex_id_from_pubmed(orig_id)
            if alex_id is None:
                alex_ids = None
            else:
                alex_ids = [alex_id]
        else:
            alex_ids = None
        if alex_ids is None:
            alex_ids = get_openalex_id_from_title(title) 
    
        alex_ids = list(map(lambda x: x.split('/')[-1], alex_ids))
        id2alex[orig_id] = alex_ids
        with open(id2alex_filename, "w") as f:
            json.dump(id2alex, f)

    di_values = list(di_scores.loc[di_scores["alex_id"].isin(alex_ids), "score"])
    if di_values:
        result[orig_id] = max(di_values)
    else:
        result[orig_id] = None

    with open("tmp_result.pickle", "wb") as f:
        pickle.dump(result, f)
    
    
result = [{ID: key, "novelty_score": value} for key, value in result.items()] 
pd.DataFrame(result).to_csv(f"{RESULTS}/di_values_10years_result.csv")


