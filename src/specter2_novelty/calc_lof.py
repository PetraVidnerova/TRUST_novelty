import sys
import os
import tqdm
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
print("Loading docvecs ... ")
files = [f for f in list(os.listdir(EMBDIR)) if f.startswith(settings.output_prefix)] 

for filename in tqdm.tqdm(files):
    rowid = filename[len(settings.output_prefix)+1:-3] # skip prefix + _ and '.pt'
    if settings.output_prefix == "arxiv_dataset":
        rowid = f"http://arxiv.org/abs/{rowid.replace(':','/')}"
        
    rowids.append(rowid)
    docvecs.append(torch.load(f"{EMBDIR}/{filename}").numpy())

docvecs = np.vstack(docvecs)
print(docvecs.shape)
print("Docvecs prepared.")

#
# TODO!!!!
# use train/reference set and testset 
#
print("Fitting LocalOutlierFactor")
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit(docvecs)
novelty_lof = clf.negative_outlier_factor_ * -1
print("Done")

print("Fitting IsolationForest")
clf = IsolationForest()
clf.fit(docvecs)
novelty_if = clf.score_samples(docvecs) * -1
print("Done")

ID = settings.id_column_name

result1 = pd.DataFrame()
result1[ID] = rowids
result1["score"] = novelty_lof 

result2 = pd.DataFrame()
result2[ID] = rowids
result2["score"] = novelty_if

result1.to_csv(f"{RESULTS}/{settings.output_prefix}_specter2_LOF_n=20_result.csv")
result2.to_csv(f"{RESULTS}/{settings.output_prefix}_specter2_IF_result.csv")
