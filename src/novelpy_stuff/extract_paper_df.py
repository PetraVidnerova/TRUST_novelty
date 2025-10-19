import json
import tqdm 
import pandas as pd

DIR = "Data/docs/Title_abs_sample"

result = []

for year in tqdm.tqdm(range(1995, 2016)):
    filename = f"{DIR}/{year}.json"
    with open(filename, "r") as f:
        data = json.load(f)
    for item in data:
        abstract_text = "\n".join([
            part["AbstractText"]
            for part in item["a04_abstract"]
        ])
        result.append(
            {
                "PMID": item["PMID"],
                "title": item["ArticleTitle"],
                "abstract": abstract_text,
                "year": item["year"]
            }
        )
        
df = pd.DataFrame(result)
print(df)
df.to_feather("novelpy_data_sample.feather")
        
