import pandas as pd
import numpy as np

class Metrics:

    def __init__(self, returns_dict):
        self.returns_dict = returns_dict
        self.metrics_df = pd.DataFrame(columns=['Cumulative Return', 'Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown'])

    def calculate(self):
        for strategy, returns in self.returns_dict.items():
            returns = returns.squeeze()
            cumulative_return = self._cumulative_return(returns)
            annual_return = self._annual_return(returns)
            annual_volatility = self._annual_volatility(returns)
            sharpe_ratio = self._sharpe_ratio(returns)
            max_drawdown = self._max_drawdown(returns)
            self.metrics_df.loc[strategy] = [cumulative_return, annual_return, annual_volatility, sharpe_ratio, max_drawdown]
        return self.metrics_df

    @staticmethod
    def _cumulative_return(returns):
        return returns.cumsum().iloc[-1]

    @staticmethod
    def _annual_return(returns):
        return returns.mean() * 252

    @staticmethod
    def _annual_volatility(returns):
        return returns.std() * np.sqrt(252)

    @staticmethod
    def _sharpe_ratio(returns, risk_free_rate=0.02):
        excess_returns = returns - risk_free_rate / 252
        return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))

    @staticmethod
    def _max_drawdown(returns):
        cum_returns = returns.cumsum()
        rolling_max = cum_returns.cummax()
        drawdown = rolling_max - cum_returns
        return drawdown.max()