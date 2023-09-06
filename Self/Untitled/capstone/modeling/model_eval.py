import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

from ..data import growth

def score(data):
    scores = ((data - data.mean()) / data.std()).sort_values(ascending=False)
    scores.name = 'score'
    return scores

def expected_return(predicted_prices):
    p_returns = (predicted_prices.iloc[-1] / predicted_prices.iloc[0]) - 1
    p_returns.name = 'e_return'
    return p_returns.sort_values(ascending=False)

def rank_stocks(predicted_prices):
    expected_returns = expected_return(predicted_prices)
    scores = score(expected_returns)
    s = pd.DataFrame([scores]).T
    ranked = s.reset_index().rename(columns={'index': 'stock'})
    ranked.index += 1
    ranked.index.name = 'rank'
    return ranked