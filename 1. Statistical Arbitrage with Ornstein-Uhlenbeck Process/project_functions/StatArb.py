import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np

# VISUALIZATIONS
# =========================================================================================================

def to_df(eigenvectors, eigenportfolios, snp_returns):
    eigenvector_df = pd.DataFrame(eigenvectors, columns=snp_returns.columns)
    eigenportfolios_df = pd.DataFrame(eigenportfolios, columns=snp_returns.columns)
    eigenvector_df.index += 1; eigenportfolios_df.index += 1
    return eigenvector_df, eigenportfolios_df

def plot_spy(spy_prices, figsize=(15,7)):
    spy_prices.plot(legend=False, figsize=figsize, linewidth=2)
    plt.title('SPY ETF Prices')
    plt.ylabel('Prices')
    plt.xlabel('')
    plt.show()

def plot_evr(evr, figsize=(15, 7), color='#C44E52'):
    plt.figure(figsize=figsize)
    sns.barplot(x=np.arange(1, len(evr)+1), y=evr, color=color, edgecolor='k')
    plt.title('Explained Variance Ratio')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Explained Variance')
    plt.show()

def plot_evr_density(evr, figsize=(15,7), color='#5D60FF'):
    plt.figure(figsize=figsize)
    sns.histplot(evr, bins=50, alpha=1, color=color, stat='percent')
    plt.title('Density Plot of Explained Variance Ratio')
    plt.xlabel('Explained Variance Ratio')
    plt.ylabel('Percentage of Principal Components')

def plot_comparative_growth(principal_eigenportfolio_cum_returns, spy_cum_returns, figsize=(15,7), color='#4273FF'):
    plt.figure(figsize=figsize)
    plot = principal_eigenportfolio_cum_returns.plot(legend=False, color=color, linewidth=2)
    spy_cum_returns.plot(ax=plot, legend=False, linestyle='dashed', linewidth=2)
    plt.axhline(y=0, color='k', linewidth=1.25, alpha=.25)
    plt.title('Cumulative Returns')
    plt.legend(['Eigen Portfolio', 'SPY ETF'])
    plt.xlabel('')
    plt.ylabel('Returns')
    plt.show()

# CORRELATION MATRIX
# =========================================================================================================

def get_correlation_matrix(snp_prices):
    snp_returns = np.log(snp_prices).diff()[1:]
    corr = snp_returns.corr()
    return snp_returns, corr

# EIGENVECTORS AND PORTFOLIOS
# =========================================================================================================

def get_eigenportfolios_raw(pca, snp_returns):
    v_i = pca.components_
    sig_i = snp_returns.std().values

    return v_i, sig_i, v_i / sig_i

def get_eigenportfolios(pca, snp_returns):
    v_i, sig_i, q_i = get_eigenportfolios_raw(pca, snp_returns)
    return v_i, sig_i, (q_i.T / np.sum(q_i, axis=1)).T


# CUMULATIVE RETURNS (EIGEN PORTFOLIO AND SPY ETF)
# =========================================================================================================

def get_eigenportfolio_returns(eigenportfolios, snp_returns):
    return (eigenportfolios @ snp_returns.T).T

def get_cum_returns(eigenportfolio_returns, spy_prices):
    principal_eigenportfolio_returns = eigenportfolio_returns[0]
    principal_eigenportfolio_cum_returns = (1 + principal_eigenportfolio_returns).cumprod() - 1
    spy_cum_returns = (1 + np.log(spy_prices).diff()).cumprod() - 1
    return principal_eigenportfolio_returns, principal_eigenportfolio_cum_returns, spy_cum_returns