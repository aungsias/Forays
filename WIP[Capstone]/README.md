# Dynamic Portfolio Optimization with Sector Rotation & Machine Learning
Aung Si<br>
September 9<sup>th</sup>, 2023
---
## Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Data](#data)
- [Feature & Target Engineering](#feature--target-engineering)
    - [Features](#features)
        - [Data Sources](#data-sources)
        - [Feature Transformation and Enhancement](#feature-transformation-and-enhancement)
        - [Statistical Tests](#statistical-tests)
        - [Dimensionality](#dimensionality)
    - [Targets](#targets)
        - [Sectoral Returns as Targets](#sectoral-returns-as-targets)
        - [Calculation](#calculation)
        - [Rationale](#rationale)
- [Modeling](#modeling)
    - [Models Used](#models-used)
        - [Regression Models](#regression-models)
        - [Time Series Models](#time-series-models)
## Abstract
In this project, I've engineered an adaptive machine learning algorithm that undergoes biannual recalibration to select the most accurate model for sector-based investment strategies. To counteract the pitfalls of over-forecasting, the algorithm employs a custom loss function that penalizes overpredictions. It comprehensively integrates a diverse range of financial indicators, including equity, debt, commodities, and market volatility. To enhance computational efficiency and model precision, I employed Principal Component Analysis for feature reduction. The model's robustness was substantiated through a 15-year backtest, during which it outperformed the SPY index by an estimated 91.85%. The finalized, vetted model has been encapsulated in a real-time dashboard, effectively translating intricate analytics into actionable market insights.

## Introduction
Financial markets are inherently volatile and self-correcting, subject to externalities including but not limited to equities, debt instruments, commodities, and market sentiment. Traditional investment models often falter in this ever-changing landscape, and even adaptable strategies like sector rotation are usually limited by heuristic-based decision-making.

In this project, I've constructed a machine learning framework that operates as a "model of models." Rather than being a singular static model, it encompasses multiple machine learning algorithms. Every six months, each constituent model is retrained and assessed for performance. The best-performing model is then selected for the ensuing period, thereby ensuring that the framework remains attuned to the current market conditions. Each model selects a sector based on the mean predicted returns—the sector with the highest return is chosen. This framework is then backtested over a 15-year period to substantiate the framework's efficacy.

## Methodology
1. **Data Retrieval** - Retrieve prices for stocks and indices via `yfinance`.
3. **Feature Engineering** - Engineered features capture essential market variables: log returns, Sharpe ratios, and lagged features. Each feature is validated for stationarity. PCA mitigates multicollinearity.
3. **Target Engineering** - Sectoral returns are the primary targets, calculated as average log returns within each GICS Sector, providing a sector-focused strategy.
4. **"Model of Models" Architecture** - Utilizes multiple machine learning models. Biannually, models are retrained and assessed. The top-performing model is chosen via a custom loss function penalizing overpredictions. The sector indicated by this model is the investment focus.
5. **Backtesting** - The selected investment strategy is backtested, accounting for transaction costs, over a historical period to validate its efficacy.

## Data

The initial data pool encompassed daily prices of all 503 constituents of the S&P 500 (the tickers for which were scraped from [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)), along with four pivotal market indices: TI, DBC, AGG, and VIX used as proxies for the prices of stocks, commodities, bonds, and volatility. All prices were fetched using the [`yfinance`](https://pypi.org/project/yfinance/) package. However, data gaps necessitated a refinement of this dataset and the analytical timeframe. Post-cleaning, the effective timeframe spanned from November 17, 2006, to September 11, 2023.

Further adjustment involved feature shifting to accommodate a 252-day forecast, equivalent to the number of trading days in a year. Consequently, the dataset's starting point for the model training was shifted to November 20, 2007. This maneuver ensured that the model's predictive variables align accurately with the forecast period, while still maintaining data integrity. This dataset provides a robust 16-year snapshot of market behavior, forming the bedrock for the machine learning framework.

## Feature & Target Engineering

### Features
The feature set is crafted to encapsulate diverse market indicators and sectoral insights. The following breaks the engineering process down:

#### Data Sources
- **S&P 500 Constituents**: 419 stocks with historical prices from February 6, 2006, to the present.
- **Market Indices**: Four key indices—Bonds, Commodities, Stocks, and Volatility—also spanning the same timeframe.
#### Feature Transformation and Enhancement
- **Log Returns**: Log returns were computed from prices to ensure stationarity of features.
- **Lookback Period**: A 10-day rolling window was used to calculate the Sharpe ratio for each feature, annualized with a factor of 252.
- **Lagged Features**: Created for each quarter (63 trading days) up to a year, capturing seasonality and longer-term trends.
#### Statistical Tests
- **ADF Test**: Augmented Dickey-Fuller test confirmed all features to be stationary, a prerequisite for time-series modeling.
#### Dimensionality
The final dataset comprises 120 features, tested for stationarity. Initially, a heatmap of feature correlations revealed significant concerns regarding multicollinearity:

<p align="center">
    <img src="img/feature_correlations.png" alt="Feature Correlations" width="30%" height="30%">
</p>

However, this issue was mitigated using Principal Component Analysis (PCA) within the modeling pipelines, details of which will be elaborated upon in the subsequent sections.

The feature set aims to capture the snapshot of current market conditions and includes backward-looking indicators that help the machine learning models understand historical market behavior. Rolling computations were employed to preclude any data leakage from future observations, thereby maintaining the integrity of the predictive models.

### Targets
#### Sectoral Returns as Targets
The primary targets for this machine learning framework are the sectoral returns, which are calculated as the average log returns of the stocks within each GICS (Global Industry Classification Standard) Sector. These sectoral returns serve as the key performance indicators that the models aim to predict, thereby enabling a sector rotation strategy.

#### Calculation
- **Sector Identification**: Stocks are first categorized into their respective GICS Sectors.
- **Log Returns**: Following identification, stock prices are transformed into log returns.
- **Average Log Returns**: For each trading day, the average log returns of the stocks in each sector are computed to generate the sectoral returns.

#### Rationale
Sectoral returns offer an aggregated, yet nuanced, view of market trends. By focusing on sectoral returns as targets, the models can capture underlying economic factors affecting specific industries. This facilitates a more informed and targeted investment strategy, compared to using broader market indices and enables a strategy that can adapt to sector-specific trends and conditions.

## Modeling
### Models Used
The machine learning framework in this project comprises an ensemble of diverse models, each with distinct strengths tailored for financial market analysis. The models are preprocessed using Principal Component Analysis (PCA) to capture at least 80% of the variance in the data, and standard scaled for normalization. Below are the models and their configurations:

#### Regression Models
- **ElasticNet**: Combines L1 and L2 regularization, aiding in feature selection and handling multicollinearity.
    - Parameters: `alpha = 1` (strong regularization), `l1_ratio = 0.5` (balanced L1 and L2)
- **Support Vector Regressor (SVR)**: Uses an RBF kernel to capture non-linear relationships.
    - Parameters: `kernel = 'rbf'`, `c = 1` (moderate regularization), `gamma = 'auto'` (automatic kernal coefficient)
- **RandomForestRegressor**: An ensemble of decision trees, capturing complex relationships and feature importance.
    - Parameters: `n_estimators = 100`
- **GradientBoostingRegressor**: Boosting algorithm suitable for capturing non-linear relationships.
    - Parameters: `n_estimators = 100`
- **XGBoostRegressor**: Optimized gradient boosting algorithm known for speed and performance
    - Parameters: `n_estimators = 100`

All regression models are configured with `random_state = 42` for reproducibility. 

#### Time Series Models
- **Naive**: Forecasts future returns based on the past six months of returns. Serves as a benchmark for performance.
- **ARIMAX**: Time series model that incorporates external variables to forecast future returns.

This ensemble enables the framework to adapt to a variety of market conditions, making it robust and versatile. The biannual recalibration process assesses the performance of these models, selecting the most effective one for the upcoming period.
