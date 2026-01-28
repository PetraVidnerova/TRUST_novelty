import pandas as pd


df = pd.read_parquet("../../data/cell/cell_papers_selected_novelty.parquet")
select = ["doi", "openalex", "title", "abstract"]

DOI = 'DOI (as URL)'
OPENALEXID = 'OpenAlexID (as URL)'

df = df[select]
print(df.columns)
df.columns = [DOI, OPENALEXID, "Title", "Abstract"]

df.info()

df[DOI] = df[DOI].apply(lambda x: f"https://doi.org/{x}")
df[OPENALEXID] = df[OPENALEXID].apply(lambda x: f"https://openalex.org/{x}")

df = df.reset_index(drop=True)
df["PaperProjectID"] = df.index


print(df)
print(df.columns)

df.to_csv("../../data/soutez/cell.csv", index=None)
