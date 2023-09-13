import pandas as pd
import pickle

def read_file(file_name, path=None, index_col=None):
    if not path:
        return pd.read_csv(f"data/{file_name}.csv", index_col=index_col, parse_dates=True)
    else:
        return pd.read_csv(f"{path}/{file_name}.csv", index_col=index_col, parse_dates=True)
    
def get_sectors():
    with open("data/sector_list.pkl", "rb") as f:
        return pickle.load(f)