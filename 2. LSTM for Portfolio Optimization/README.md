# Long Short-Term Memory for Portfolio Optimization

---

## Contents

- [Introduction](#introduction)
  - [Key Points in the Study](#key-points-in-the-study)
  - [Disclaimer](#disclaimer)
- [Problem Statement](#problem-statement)
- [Implementation](#implementation)
  - [Data](#data)
  - [Model Architecture](#model-architecture)
  - [Objective Function](#objective-function)
  - [Training Scheme](#training-scheme)
  - [Setting up the Backtest](#setting-up-the-backtest)
- [The Backtest](#the-backtest)
  - [Performance Evaluation](#performance-evaluation)
  - [Adaptability to Market Turmoil](#adaptability-to-market-turmoil)

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

### Data

I use the same four indices to evaluate the LSTM model, but use a slightly different timeframe than did the authors (2006 to 2020 in the paper versus 2007 to 2023 in my replication). The prices for the four indices were retrieved using the `yfinance` library and can be accessed [here](data/VTI-AGG-DBC-VIX.csv) and loaded in with `pandas`.

Logarithmic returns are computed of off the prices and the two dataframes are concatenated as such:

```python
import pandas as pd
import numpy as np

prices = pd.read_csv('data/VTI-AGG-DBC-VIX-prices.csv', index_col='Date', parse_dates=True)
returns = np.log(prices).diff()[1:]
data = pd.concat([prices.loc[returns.index], returns], axis=1)
```

### Model Architecture

The model proposed in the paper consists of three main components: the input layer, the neural layer, and the output layer:

- **Input Layer**: The model takes in information from multiple assets $A_{i}$ to form a portfolio. Each asset's features, such as past prices and returns, are concatenated, resulting in an input dimension of $(k, 2 \times n)$, where $k$ is the lookback window and $n$ is the number of assets.

- **Neural Layer**: Various deep learning architectures, including fully connected neural networks (FCN), convolutional neural networks (CNN), and Long Short-Term Memory networks (LSTM), were tested. LSTM demonstrated the best performance for daily financial data due to its ability to filter and summarize information efficiently, resulting in better generalization. FCNs were prone to overfitting, while CNNs sometimes led to underfitting.

- **Output Layer**: To build a long-only portfolio, a softmax activation function is used, ensuring the portfolio weights are positive and sum to one. The output nodes correspond to the number of assets, and the portfolio weights are multiplied with the assets' returns to calculate the realized portfolio returns. The Sharpe ratio is then derived, and gradient ascent is applied to update the model parameters

For simplicity, I've implemented the model as such:

```python
import torch
from torch.nn import LSTM, Linear, Module

class PortOptNN(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PortOptNN, self).__init__()
        self.lstm = LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = Linear(hidden_dim, output_dim, bias=True)
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out_last = lstm_out[:, -1, :]
        fc_out = self.fc(lstm_out_last)
        output = self.softmax(fc_out)
        return output
```

The key difference is the focus on the LSTM architecture. While the paper experimented with various deep learning models (FCNs and CNNs), I specifically test an LSTM as the paper found it to have delivered the best performance in terms of 1) finding weights that maximize the Sharpe Ratio 2) resilience in the face of market turmoil, and 3) providing the highest cumulative return net fees.

### Objective Function
The core objective of the model is to maximize the Sharpe ratio, a widely recognized measure of risk-adjusted returns. The Sharpe ratio quantifies the return per unit of risk in a portfolio and in effect measures the ability of the portfolio to generate returns considering its underlying risk. The original paper circumvents the forecasting of future returns and maximizes the following function directly for each trading period:

$$L_{T} = \frac{E(R_{p,t})}{\sqrt{E(R^2_{p,t}) - (E(R_{p,t}))^2}}$$
$$E(R_{p,t}) = \frac{1}{T}\sum^T_{t=1}R_{p,t}$$

where $R_{p,t}$ is realized portfolio return over $n$ assets at time $t$:

```python
def objective(outputs, targets):
    portfolio_returns = (outputs * targets).sum(dim=1)
    mean_portfolio_return = portfolio_returns.mean()
    volatility = torch.std(portfolio_returns)
    sharpe_ratio = mean_portfolio_return / volatility
    return sharpe_ratio
```

### Training Scheme

Our LSTM model is trained over 100 epochs, where each epoch represents a complete pass through the training dataset. The model's predictions (portfolio weights) are passed through an objective function to compute the gain of the daily Sharpe Ratio. The Adam optimizer is used to update the model's parameters in the direction that maximizes this gain (note that by setting `maximize` to `True`, we are using gradient ascent).

The data is divided into training and testing sets, with the first half used for training and the second half for evaluation. During training, the gradients are computed based on the gain, and the optimizer updates the model's weights accordingly:

```python

seq_len = 50 # 50 day lookback

n_assets = len(returns.columns)
n_samples = len(data[seq_len:])
n_features = len(data.columns)

input_data = np.zeros((n_samples, seq_len, n_features)) # (n_samples, seq_len, num_features)
target = returns.iloc[seq_len:].values[:n_samples] # (n_samples, n_assets)

for i in range(n_samples):
    input_data[i] = features.iloc[i:(i + seq_len)].values

input_dim = input_data.shape[-1] 
hidden_dim = 64
output_dim = target.shape[-1]

model = PortOptNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), maximize=True) # maximization of objective function

train_input = input_data[:len(input_data)//2]
train_input_torch = torch.tensor(train_input, dtype=torch.float32)

train_target = target[:len(target)//2]
train_target_torch = torch.tensor(train_target, dtype=torch.float32)

test_input = input_data[len(input_data)//2:len(input_data)]
test_input_torch = torch.tensor(test_input, dtype=torch.float32)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_input_torch)
    gain = objective(outputs, train_target_torch)
    gain.backward()
    optimizer.step()

with torch.no_grad():
    test_outputs = model(test_input_torch)
```

Below is how the gains progressed over the 100 epochs:

![Evolution of gains over 100 epochs](img/gains.png)

After training, the model's performance can be evaluated on the testing set.

### Setting up the Backtest

The weights predicted by the model represent the desired allocation of assets in the portfolio for future periods. However, in a real-life trading scenario, decisions must be made based on information available at the current time or earlier. Therefore, to mimic real-life investment behavior in the backtest, the predicted weights must be shifted appropriately. This ensures that the backtest accounts for the necessary lead time to make trading decisions and execute orders, reflecting the inherent delays and information lags that are part of actual trading. The authors thus denote the computation of portfolio returns using the model's outputs as:

$$R_{p,t} = \sum^n_{i=1}w_{i,t-1} \cdot r_{i,t}$$

```python
weights = pd.DataFrame(
    test_outputs, 
    index=returns.index[-len(test_outputs):],
    columns=returns.columns
).shift()

backtest_returns = returns.loc[weights.index].copy()
backtest_returns.loc[backtest_returns.index.min()] = {stock: 0 for stock in indices}
model_returns = weights.multiply(backtest_returns).sum(axis=1)
```

---

## The Backtest

The backtest below was conducted after accounting for transaction costs as per:

$$C_{t} = \text{r} \times \sum^n_{i=1}|w_{i,t} - w_{i,t-1}|$$

where $\text{r}$ is the transaction cost rate (I test both 0.1% and 1%), and the term on the right is the absolute difference in weights day to day. Thus the portfolio returns net fees is given by:

$$R_{p,t} - C_{t}$$

With that, the backtest is broken down into the following components

1. Comparison of cumulative log returns (net fees) of the LSTM model versus the four indices VTI, DBC, AGG, and ^VIX.
2. Evaluation of the annual return, annual volatility, Sharpe Ratio, maximum drawdown of each asset.

### Performance Evaluation

![Cumulative returns](img/cumulative_returns.png)

We can see that even without a walk-forward backtest and even with transaction costs as high as 1%—i.e. the LSTM model was trained on the first half of the dataset and tested on the latter half—it outpeformed all the individual indices.

Below are the respective metrics for each investment within our backtest:
![Investment metrics](img/metrics.png)

### Adaptability to Market Turmoil

![Allocations during first quarter of 2020](img/allocations.png)

Though our training scheme differed from that of the study, the above shows that the LSTM model was still adapative to market turmoil. During the volatile first quarter of 2020, marked by significant market turmoil, the adaptability of the LSTM model was put to the test. Although the training scheme for this model differed from the referenced study, the results showed similar resilience.

The LSTM model navigated the uncertainty of this period; starting in March 2020, the model began shifting its portfolio allocations primarily towards bonds. This is a notable decision, as bonds are often seen as a safer asset class, particularly during economic downturns. The model thus aimed to limit the portfolio's exposure to the higher volatility present in other investment options. This adaptive behavior underscores the ability the LSTM model to detect and respond to complex market conditions. 