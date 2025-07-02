import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

train = pd.read_csv("data_petra/papers_train.csv", index_col=0)
test = pd.read_csv("data_petra/papers_test.csv", index_col=0)

llm_train = pd.read_csv("data_petra/papers_train_llm.csv", index_col=1)
llm_test = pd.read_csv("data_petra/papers_test_llm.csv", index_col=1)

train["llm_novelty"] = llm_train["llm_novelty"]
test["llm_novelty"] = llm_test["llm_novelty"]

select_train = (train["year"] > 2001).values
select_test = (test["year"] > 2001).values


train = train[select_train]
test = test[select_test]

llm_train = llm_train[select_train]
llm_test = llm_test[select_test]

print("Trainset length:", len(train))
print("Testset length:", len(test))

def calculate_novelties(in_train, in_test, out_train=None, out_test=None, method="LOF"):

    if method == "LOF":
        est = LocalOutlierFactor(n_neighbors=100, novelty=True)
        est.fit(in_train)
        novelty = est.decision_function(in_test)

    elif method == "IF":
        est = IsolationForest()
        est.fit(in_train)
        novelty = est.decision_function(in_test)

    elif method == "XGBC":
        est  = XGBClassifier(use_label_encoder=False, n_estimators=4)
        est.fit(in_train, out_train)
        novelty = est.predict_proba(in_test)[:, 1]
        
    elif method == "XGBR":
        est  = XGBRegressor(n_estimators=4)
        est.fit(in_train, out_train)
        novelty = est.predict(in_test)

    return novelty

method = "XGBR"
features  = ["year", "forward citations", "backward citations"]
in_train = train[features]
in_test = test[features]
out_train = train["NEW_FINDING"]
out_test = test["NEW_FINDING"]

novelty_scores = calculate_novelties(in_train, in_test, out_train, out_test, method=method) 
test["XGBR"] = novelty_scores

method = "XGBC"
out_train = (train["NEW_FINDING"] > 0).astype(int)
out_test = (test["NEW_FINDING"] > 0).astype(int)
novelty_scores = calculate_novelties(in_train, in_test, out_train, out_test, method=method)
test["XGBC"] = novelty_scores

method = "XGBC"
features  = ["year", "forward citations", "backward citations", "llm_novelty"]
in_train = train[features]
in_test = test[features]
in_train["llm_score"] = llm_train["llm_novelty"]
in_test["llm_score"] = llm_test["llm_novelty"]
novelty_scores = calculate_novelties(in_train, in_test, out_train, out_test, method=method)
test["XGBC+"] = novelty_scores

method = "XGBR"
out_train = train["NEW_FINDING"]
out_test = test["NEW_FINDING"]
novelty_scores = calculate_novelties(in_train, in_test, out_train, out_test, method=method)
test["XGBR+"] = novelty_scores


# load docvectors
""" 
test["fastText + LOF"] = None
test["fastText + IF"] = None
all_docvectors = []
for i in range(1974, 2022):
    print("Loading docvecs for year:", i) 
    docvectors = np.load(f"data/docvecs/fastText/docvecs200_year={i}.npy")
    all_docvectors.append(docvectors)

    method = "LOF"
    train_selection = train.index[train["year"] == i]
    test_selection = test.index[test["year"] == i]
    if len(train_selection) == 0 or len(test_selection) == 0:
        print(f"No data for year {i}, skipping...")
        continue
    in_train = docvectors[train_selection] 
    in_test = docvectors[test_selection]
    novelty_scores = calculate_novelties(in_train, in_test, method=method)
    test.loc[test_selection, "fastText + LOF"] = novelty_scores

    method = "IF"
    novelty_scores = calculate_novelties(in_train, in_test, method=method)
    test.loc[test_selection, "fastText + IF"] = novelty_scores
 """
print("Loading BERT docvectors...")

# BERT 
in_train = np.load(f"data_petra/bert_title_train.npy")[select_train,:]
in_test = np.load(f"data_petra/bert_title_test.npy")[select_test,:]
out_train = (train["NEW_FINDING"] > 0).astype(int)
out_test = (test["NEW_FINDING"] > 0).astype(int)

method = "LOF"
novelty_scores = calculate_novelties(in_train, in_test, method=method)
test["BERT LOF"] = novelty_scores
method = "IF"
novelty_scores = calculate_novelties(in_train, in_test, method=method)
test["BERT IF"] = novelty_scores
method = "XGBC"
novelty_scores = calculate_novelties(in_train, in_test,
                                     out_train, out_test, method=method)
test["BERT XGBC"] = novelty_scores

print("Loading BERT+ docvectors...")

in_train = np.load(f"data_petra/bert_abstract_train.npy")[select_train,:]
in_test = np.load(f"data_petra/bert_abstract_test.npy")[select_test,:]
method = "LOF"
novelty_scores = calculate_novelties(in_train, in_test, method=method)
test["BERT_a LOF"] = novelty_scores
method = "IF"
novelty_scores = calculate_novelties(in_train, in_test, method=method)
test["BERT_a IF"] = novelty_scores
method = "XGBC"
novelty_scores = calculate_novelties(in_train, in_test,
                                     out_train, out_test, method=method)
test["BERT_a XGBC"] = novelty_scores

# ELMO
in_train = np.load(f"data_petra/elmo_title_train.npy")[select_train,:]
in_test = np.load(f"data_petra/elmo_title_test.npy")[select_test,:]
out_train = (train["NEW_FINDING"] > 0).astype(int)
out_test = (test["NEW_FINDING"] > 0).astype(int)

method = "LOF"
novelty_scores = calculate_novelties(in_train, in_test, method=method)
test["ELMO LOF"] = novelty_scores
method = "IF"
novelty_scores = calculate_novelties(in_train, in_test, method=method)
test["ELMO IF"] = novelty_scores
method = "XGBC"
novelty_scores = calculate_novelties(in_train, in_test,
                                     out_train, out_test, method=method)
test["ELMO XGBC"] = novelty_scores

print("Loading ELMO+ docvectors...")

in_train = np.load(f"data_petra/elmo_abstract_train.npy")[select_train,:]
in_test = np.load(f"data_petra/elmo_abstract_test.npy")[select_test,:]
method = "LOF"
novelty_scores = calculate_novelties(in_train, in_test, method=method)
test["ELMO_a LOF"] = novelty_scores
method = "IF"
novelty_scores = calculate_novelties(in_train, in_test, method=method)
test["ELMO_a IF"] = novelty_scores
method = "XGBC"
novelty_scores = calculate_novelties(in_train, in_test,
                                     out_train, out_test, method=method)
test["ELMO_a XGBC"] = novelty_scores

import torch
net = torch.load("elmo_t.pt", map_location=torch.device('cpu'))
in_test = np.load(f"data_petra/elmo_title_test.npy")[select_test,:]
novelty_scores = []
for input in in_test:
    input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        novelty_scores.append(net(input)[:, 1].item())
test["ELMO NN"] = novelty_scores

net = torch.load("elmo_a.pt", map_location=torch.device('cpu'))
in_test = np.load(f"data_petra/elmo_abstract_test.npy")[select_test,:]
novelty_scores = []
for input in in_test:
    input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        novelty_scores.append(net(input)[:, 1].item())
test["ELMO_a NN"] = novelty_scores


test.to_csv("data_petra/papers_test_novelty_scores.csv")
print('Saved.')