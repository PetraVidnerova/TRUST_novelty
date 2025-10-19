import pandas as pd


df = pd.read_csv("data_petra/papers_all_tags.csv", index_col=0)

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['year'])
train.to_csv("data_petra/papers_train.csv")
test.to_csv("data_petra/papers_test.csv")
print("Train and test sets created and saved.")

