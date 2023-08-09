# Long Short-Term Memory for Portfolio Optimization

---

## Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Prediction and Rebalancing](#prediction-and-rebalancing)
- [Backtesting and Results](#backtesting-and-results)
- [Conclusion](#conclusion)

---

## Introduction
This project is inspired by the research paper "Deep Learning for Portfolio Optimization" by Zihao Zhang, Stefan Zohren, and Stephen Roberts from the Oxford-Man Institute of Quantitative Finance, University of Oxford. The paper presents an unorthodox approach to portfolio optimization that leverages deep learning models to directly optimize the portfolio Sharpe ratio.

### Key Points in the Study
- **Circumventing Forecasting Requirements**: Traditional portfolio optimization often relies on forecasting expected returns. This study, however, introduces a framework that eliminates the need for such forecasting. It directly optimizes portfolio weights by updating model parameters, providing a novel means for asset allocation.

- **Trading Exchange-Traded Funds (ETFs)**: Instead of selecting individual assets, the paper focuses on trading ETFs of market indices to form a portfolio. This approach simplifies the asset selection process by way of utilising market indices as proxies for larger universes of asset classes.

- **Performance Evaluation**: The paper extensively compares the proposed method with various algorithms. The results demonstrate superior performance over the testing period, from 2011 to the end of April 2020. This includes periods of financial instability, highlighting the model's robustness.

- **Sensitivity Analysis**: An insightful sensitivity analysis is conducted to understand the relevance of input features, shedding light on the critical factors influencing the model's success.

This project aims to emulate the study's methodology and apply it to a new dataset, exploring the practical implementation of deep learning in the context of portfolio optimization. The motivation behind this emulation is to provide a readily implementable public codebase for usage, scrutiny, and education. The [accompanying reference paper](<./[REFERENCE PAPER] Deep Learning for Portoflio Optimization, University of Oxford.pdf>) provides a comprehensive understanding of the underlying theory and techniques.

### Disclaimer
Please note that while this project seeks to faithfully emulate the methodology described in the accompanying reference paper, certain complexities and nuances may have been simplified or omitted for the sake of accessibility and clarity. This implementation should be seen as an educational resource and a starting point for further exploration rather than a complete replication of the original study's full sophistication. Users are encouraged to refer to the original paper for a comprehensive understanding of the underlying theory and techniques.

---

## Problem Statement

Predict optimal weights for assets in a portfolio given historical price and daily return data so as to maximize the portfolio's Sharpe Ratio.

---

## Methodological Changes and Breakdown

Below is the breakdown of how I've replicated the study. I flag the steps that are different from the study with a ***D***, and those that align with an ***S***.

| Step | Description                                                                                   | Alignment |
|------|-----------------------------------------------------------------------------------------------|-----------|
| 1    | Chosen tickers: 'VTI' (Vanguard Total Stock Market ETF), 'AGG' (iShares Core U.S. Aggregate Bond ETF), 'DBC' (Invesco DB Commodity Index Tracking Fund), '^VIX' (CBOE Volatility Index). | S         |
| 2    | Chosen timeframe: January 1<sup>st</sup>, 2007 to August 7<sup>th</sup>, 2023.               | D         |
| 3    | Compute log returns.                                                                          | S         |
| 4    | Concatenate prices and log returns to get an aggregate features dataset.                      | S         |
| 5    | Defining the architecture of the model (explored later).                                                 | D         |
| 6    | Set the first half of the features dataset as training period, and the latter half as the testing period. | D         |
| 7    | Backtest.                                                                                     | S         |
---

## Implementation

First we need the following libraries and modules. We build our model with [PyTorch](https://pytorch.org/) and use [yfinance](https://pypi.org/project/yfinance/) to retrieve price data. `trade_metrics` is a package that I wrote, which you can find [here](https://github.com/aungsias/Eigen/tree/main/Custom%20Packages/trade_metrics) - it is used to compute the statistics of an asset or portfolio such as Cumulative Log Returns, Sharpe Ratio, and more.

```
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from trade_metrics.metrics import Metrics
from torch.nn import LSTM, Linear, Module, Softmax
from scipy.optimize import minimize
from tqdm.auto import tqdm
```

Now we retrieve the data for the four indices and compute log returns, and concatenate the prices and the returns into one `features` dataframe:

```
start = '2007-01-01'
end = '2023-08-07'

indices = ['VTI', 'AGG', 'DBC', '^VIX']
prices = yf.download(indices, start=start, end=end)['Close'].dropna(axis=1)
returns = np.log(prices).diff()[1:]

features = pd.concat([prices.loc[returns.index], returns], axis=1)
features.head()
```
![First 5 rows of `features` dataframe](img/features_head.png)
