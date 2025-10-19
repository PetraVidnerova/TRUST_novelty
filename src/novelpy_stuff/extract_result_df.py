import json
import tqdm 
import pandas as pd

DIR = "Result"
METHODS = ["foster"]



for method in METHODS:
    result = []
    for year in tqdm.tqdm(range(2000, 2011)):
        filename = f"{DIR}/{method}/c04_referencelist/{year}.json"
        
        with open(filename, "r") as f:
            data = json.load(f)

        for item in data:
            result.append({
                "PMID": item["PMID"],
                "score": item[f"c04_referencelist_{method}"]["score"]["novelty"]
            })

    df = pd.DataFrame(result)
    print(df)
    df.to_feather(f"novelpy_{method}_result.feather")
        
