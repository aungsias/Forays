import pandas as pd

def read_file(path, name, index_col=None):
    return pd.read_csv(f"{path}/{name}.csv", index_col=index_col, parse_dates=True)