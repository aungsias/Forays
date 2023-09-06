import requests
import json
import pandas as pd
import random
import yfinance as yf

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
    )[col]

def fetch_fundamentals(ticker, api_key, limit=130, period='quarter', metrics=None):
    url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period={period}&limit={limit}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        key_metrics_data = json.loads(response.text)
        key_metrics_df = pd.DataFrame(key_metrics_data).set_index('date').iloc[::-1]
        if metrics is None:
            return key_metrics_df
        else:
            return key_metrics_df[metrics]
    else:
        print(f"Failed to fetch data for {ticker}")
        return None