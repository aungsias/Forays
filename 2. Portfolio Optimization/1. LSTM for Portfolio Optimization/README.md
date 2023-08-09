# Portfolio Optimization Neural Network

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

## Implementation

First we need the following libraries and modules

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

The dataset consists of historical price data for specified indices, including 'VTI', 'AGG', 'DBC', '^VIX'. The data is retrieved from Yahoo Finance and transformed into logarithmic returns and combined with price information to form the feature set.

### Features

- **Prices**: Historical closing prices for the specified indices.
- **Returns**: Logarithmic daily returns calculated from the prices.

## Model Architecture

The core of the model consists of the following layers:

1. **LSTM Layer**: Processes time-series data, capturing temporal dependencies within the data. It's configured with a specific number of hidden dimensions, controlling the complexity of the model.
2. **Linear Layer**: Transforms the LSTM output to the desired output dimension, allowing the model to make predictions in the form of asset weights.
3. **Softmax Layer**: Applies a softmax activation to ensure that the output represents valid weight allocations, summing to one.

## Training Process

The training process involves the following steps:

1. **Data Preparation**: Sequences of data are prepared with a defined length, creating 3D input arrays for the LSTM.
2. **Objective Function**: A custom objective function is designed to optimize the Sharpe ratio or other portfolio metrics.
3. **Training Loop**: The model is trained iteratively using the Adam optimizer, and the progress is monitored through a custom gain metric.

## Prediction and Rebalancing

Once trained, the model is used to predict asset weights for the test data. The weights can be used to execute trades and rebalance the portfolio periodically.

## Backtesting and Results

Backtesting is performed to evaluate the model's performance in a simulated trading environment. Different scenarios, including various transaction costs, can be evaluated.

### Plots

- **Cumulative Log Returns**: Visualization of cumulative log returns over time, compared with benchmark strategies.
- **Asset Allocation**: Visualization of the model's predicted asset allocations over time.

## Conclusion

This project demonstrates a novel approach to portfolio optimization using deep learning. By incorporating LSTM networks and custom training objectives, it provides a data-driven way to make investment decisions. The methodology can be extended to other financial applications and enhanced with additional features and optimization techniques.