import pandas as pd

def read_df(name, *args, **kwargs):
    if name.endswith(".feather"):
        return pd.read_feather(name)
    elif name.endswith(".parquet"):
        return pd.read_parquet(name, engine="fastparquet")
    else:
        return pd.read_csv(name, **kwargs)
