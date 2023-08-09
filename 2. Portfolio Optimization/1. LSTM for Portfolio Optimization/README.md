# Portfolio Optimization Neural Network

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Prediction and Rebalancing](#prediction-and-rebalancing)
- [Backtesting and Results](#backtesting-and-results)
- [Conclusion](#conclusion)

## Introduction

This project focuses on leveraging deep learning to optimize portfolio allocations among various financial assets. It provides a systematic way to allocate capital among different investment opportunities to achieve a specific objective, such as maximizing returns or the Sharpe ratio.

## Problem Statement

The challenge is to predict optimal weights for assets in a portfolio based on historical data. The goal is to maximize the portfolio's Sharpe ratio, subject to constraints such as weights summing to one.

## Dataset

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