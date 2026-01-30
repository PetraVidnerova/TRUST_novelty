import pickle
import pandas as pd


with open("tmp_result_cellref.pkl", "rb") as f:
    result = pickle.load(f)

df = pd.DataFrame.from_dict(result, orient='index')
df["PaperProjectID"] = df.index
print(df)

df = df[df['titles_only'] == False]
#df = df[df['n_related'] >= 9]

df_orig = pd.read_csv("../../data/soutez/cell.csv")
print(df_orig)

print(df.columns)
print(df_orig.columns)

df_merged = df.merge(df_orig, on="PaperProjectID", how="left", suffixes=('_eval', '_orig'))
df_merged = df_merged[["PaperProjectID", "OpenAlexID (as URL)", "score"]]

print(df_merged)
print(df_merged.columns)

df_merged.to_csv("cell_eval_results_abstracts_only.csv", index=False)
