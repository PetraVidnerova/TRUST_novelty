from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import torch
import numpy as np
import pandas as pd

docvecs = []
for i in range(5011):
    docvecs.append(torch.load(f"emb/{i}.pt").detach().cpu().numpy())

docvecs = np.vstack(docvecs)
print(docvecs.shape)

clf = LocalOutlierFactor(n_neighbors=10)
clf.fit(docvecs)
novelty_lof = clf.negative_outlier_factor_ * -1

clf = IsolationForest()
clf.fit(docvecs)
novelty_if = clf.score_samples(docvecs) * -1

df = pd.read_csv("../arxiv_dataset_all_info.csv")

result1 = pd.DataFrame()
result1["id"] = df["id"]
result1["score"] = novelty_lof 

result2 = pd.DataFrame()
result2["id"] = df["id"]
result2["score"] = novelty_if

result1.to_csv("specter2_LOF.csv")
result2.to_csv("specter2_IF.csv")
