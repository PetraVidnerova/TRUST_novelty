import sys
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import torch
import numpy as np
import pandas as pd

from utils.config import create_settings

DATADIR = "../../data"
EMBDIR = f"{DATADIR}/emb"
RESULTS = "../../results"

if len(sys.argv) > 1:
    settings = create_settings(sys.argv[1])
else:
    settings = create_settings(".env")

rowids = [] 
docvecs = []
for filename in os.listdir(EMBDIR):
    if not filename.startswith(settings.output_prefix):
        continue
    rowid = filename[len(settings.output_prefix)+1:-3] # skip prefix + _ and '.pt'
    if settings.output_prefix == "arxiv_dataset":
        rowid = f"http://arxiv.org/abs/{rowid.replace(':','/')}"
        
    rowids.append(rowid)
    docvecs.append(torch.load(f"{EMBDIR}/{filename}").numpy())

docvecs = np.vstack(docvecs)
print(docvecs.shape)

clf = LocalOutlierFactor(n_neighbors=10)
clf.fit(docvecs)
novelty_lof = clf.negative_outlier_factor_ * -1

clf = IsolationForest()
clf.fit(docvecs)
novelty_if = clf.score_samples(docvecs) * -1


result1 = pd.DataFrame()
result1["id"] = rowids
result1["score"] = novelty_lof 

result2 = pd.DataFrame()
result2["id"] = rowids
result2["score"] = novelty_if

result1.to_csv(f"{RESULTS}/{settings.output_prefix}_specter2_LOF_result.csv")
result2.to_csv(f"{RESULTS}/{settings.output_prefix}_specter2_IF_result.csv")
