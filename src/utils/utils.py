import pandas as pd

def read_df(name, *args, **kwargs):
    if name.endswith(".feather"):
        return pd.read_feather(name)
    else:
        return pd.read_csv(name, **kwargs)
