import os
import sys 
import json
import tqdm
import pandas as pd
import logging 
import pyalex

pyalex.config.email = "petra@cs.cas.cz"

pyalex.config.max_retries = 2
pyalex.config.retry_backoff_factor = 0.1
pyalex.config.retry_http_codes = [429, 500, 503]

def clean_title(title):
    title = title.replace(",", "").replace(".", "").replace("\n", " ")
    title = title.replace("-", "").replace("!", "")
    title = title.replace("&amp;", "").replace("&", "")
    title = " ".join(title.split())
    return title.lower()

def get_openalex_id_from_arxiv(arxiv_url):

    number = arxiv_url[len("http://arxiv.org/abs/"):]
    if "v" in number:
        number = number[:number.index("v")]
    for query in f"http://arxiv.org/abs/{number}", f"https://arxiv.org/abs/{number}":

        ws = pyalex.Works().filter(**{  # "primary_location.source.id": ID,
            "primary_location.landing_page_url": query
        }).get()

        if len(ws) > 0:
            return [w['id'] for w in ws]  
    return None

def get_openalex_id_from_title(title):
    collection = pyalex.Works().search_filter(title=clean_title(title)).get()
    results = []
    for paper in collection:
        if clean_title(paper["title"]) != clean_title(title):
            continue
        
        results.append(paper["id"])
    return results


df = pd.read_csv("arxiv_dataset_all_info.csv")
# di_scores = pd.read_csv("DI_data/OpenAlexID_Year_DindexTenYears.txt.gz", sep="\t", header=None)
# di_scores.columns = ["alex_id", "year", "score"]

# di_scores.drop(columns=["year"]).to_feather("DI_data/OpenAlexID_Year_DindexTenYears.feather")
# exit()



di_scores = pd.read_feather("DI_data/OpenAlexID_Year_DindexTenYears.feather")

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
pd.DataFrame(result).to_csv("di_values_10years_result.csv")


