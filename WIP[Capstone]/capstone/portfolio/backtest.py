import pandas as pd

def __backtest_portfolio__(allocation_df, constituent_returns):
    """
    Backtest a single portfolio based on asset allocation and constituent returns.
    
    Parameters:
    - allocation_df (DataFrame): Asset allocation for the portfolio.
    - constituent_returns (DataFrame): Returns for each constituent asset.
    
    Returns:
    - DataFrame: Cumulative returns of the backtested portfolio.
    """
    
    # Align index of allocation DataFrame with that of the constituent returns DataFrame
    strat_rets = allocation_df.reindex(constituent_returns.index).ffill().dropna()
    
    # Multiply each asset's allocation with its returns
    strat_rets = strat_rets * constituent_returns.reindex(strat_rets.index)
    
    # Sum across all assets to get portfolio return at each time step
    strat_rets = strat_rets.sum(axis=1)
    
    # Set return to zero at the starting point
    strat_rets.loc[strat_rets.index.min()] = 0
    
    return strat_rets

def backtest_portfolios(*allocation_dfs, constituent_returns):
    """
    Backtest multiple portfolios based on their asset allocations and constituent returns.
    
    Parameters:
    - *allocation_dfs (DataFrame): Asset allocations for multiple portfolios.
    - constituent_returns (DataFrame): Returns for each constituent asset.
    
    Returns:
    - tuple: Cumulative returns of all backtested portfolios.
    """
    
    strat_ret_dfs = []
    
    # Iterate through each portfolio's asset allocation DataFrame
    for df in allocation_dfs:
        # Backtest the portfolio and store its cumulative returns
        strat_ret_dfs.append(__backtest_portfolio__(df, constituent_returns))
        
    return tuple(strat_ret_dfs)
