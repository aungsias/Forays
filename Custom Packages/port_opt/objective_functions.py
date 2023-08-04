import numpy as np
import pandas as pd

def variance(weights, prices):
    returns = np.log(prices).diff()[1:]
    cov = returns.cov() * 252
    return weights @ cov @ weights.T

def sharpe_ratio(weights, prices):
    returns = np.log(prices).diff()[1:]
    portfolio_returns = returns @ weights.T
    expected_return = np.mean(portfolio_returns) * 252
    portfolio_std = np.sqrt(variance(weights, prices))
    return (expected_return) / portfolio_std 

def neg_sharpe_ratio(weights, prices):
    return -sharpe_ratio(weights, prices)

def risk_contributions(weights, prices):
    returns = np.log(prices).diff()[1:]
    cov = returns.cov() * 252
    return (weights *(cov @ weights.T)) / variance(weights, prices)

def risk_parity_obj(weights, prices):
    num_assets = len(prices.columns)
    asset_contributions = risk_contributions(weights, prices)
    equal_contribution = 1.0 / num_assets
    risk_parity_obj = np.sum((asset_contributions - equal_contribution)**2)
    return risk_parity_obj