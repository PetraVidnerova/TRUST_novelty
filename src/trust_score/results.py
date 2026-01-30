import pandas as pd
import numpy as np

df = pd.read_csv("cell_eval_results_abstracts_only.csv")
df_gold = pd.read_parquet("../../data/cell/cell_papers_all_novelty_scores.parquet")


df.columns = ['PaperProjectID', 'openalex', 'TrustScore']
df["openalex"] = df["openalex"].apply(lambda x: x[len("https://openalex.org/"):])



df_gold.info()


all_df = df.merge(df_gold, on='openalex', how='left')
all_df


all_df.info()


all_df = all_df[df['TrustScore'] != -1]


score_columns = [col for col in all_df.columns if col not in ['PaperProjectID', 'openalex', 'pubmed_id']]


corr = all_df[score_columns].corr(numeric_only=True, method='spearman')



print(corr)

