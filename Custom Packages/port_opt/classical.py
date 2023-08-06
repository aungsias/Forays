import omega as o
import numpy as np
import pandas as pd

from .objective_functions import *
from scipy.optimize import minimize

class ClassicOptimizer:

    def __init__(self, prices, shorts=False, method='max-sharpe'):
        self.stocks = prices.columns.to_list()
        self.prices = prices
        self.shorts = shorts
        self.bounds = [(-1, 1) if self.shorts else (0, 1) for _ in range(len(self.stocks))]
        self.constraints = ({'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1})
        self.initial_weights = np.array([1/len(self.stocks)] * len(self.stocks))
        self.methods = {
            'max-sharpe': neg_sharpe_ratio, 
            'min-variance': variance, 
            'risk-parity': risk_parity_obj
        }
        self.objective = self.methods[method]

    def serialize(self):
        return pd.Series(self.weights, index=self.stocks)

    def optimize(self):

        opt = minimize(
            lambda w: self.objective(weights=w, prices=self.prices),
            self.initial_weights,
            bounds=self.bounds,
            constraints=self.constraints,
        )

        self.weights = opt.x
        self.sharpe = sharpe_ratio(self.weights, self.prices)
        self.risk = np.sqrt(variance(self.weights, self.prices))
        self.risk_contributions = risk_contributions(self.weights, self.prices)