import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import random

def fetch_snp_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    symbols = table[0]['Symbol'].to_list()
    return [
        symbol.replace('.', '-') for symbol in symbols
    ]

def snp_tickers_random(random_state=None, sample=25):
    random.seed(random_state)
    return random.sample(fetch_snp_tickers(), sample)

def fetch_stock_data(tickers, start, end, col='Close', progress=False):
    return yf.download(
        tickers, start, end, progress=progress
    )[col].dropna(axis=1, how='any')

def fetch_economic_data(indicators, start, end, api_key=None):
    economic_data = {}
    
    for indicator in indicators:
        try:
            data = web.DataReader(indicator, "fred", start, end, api_key=api_key)
            economic_data[indicator] = data
        except Exception as e:
            print(f"Could not fetch data for {indicator}. Error: {e}")
            
    return economic_data

def ts_train_test_split(X, y, test_size = 1/3):
    len_test = int(len(X) * test_size)
    X_train, X_test = X[:-len_test], X[-len_test:]
    y_train, y_test = y[:-len_test], y[-len_test:]
    return X_train, X_test, y_train, y_test