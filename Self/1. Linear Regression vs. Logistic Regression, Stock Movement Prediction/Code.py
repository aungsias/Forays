import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('whitegrid')
sns.set_palette('Set1')
plt.rcParams['figure.figsize'] = (15, 7)

stock = ['AAPL']
indices = ['VTI', 'DBC', '^VIX', 'AGG']

start = '2022-01-01'
end = '2023-08-14'

prices = yf.download(stock + indices, start, end, progress=False)['Close']
data = np.log(prices).diff()[1:]
data.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

X = data[indices]
y_linear = data[stock].squeeze()
y_logistic = np.sign(y_linear)

X_train, X_test, y_train_linear, y_test_linear = train_test_split(X, y_linear, shuffle=False, test_size=.2, random_state=42)
_, _, y_train_logistic, y_test_logistic = train_test_split(X, y_logistic, shuffle=False, test_size=.2, random_state=42)

from sklearn.linear_model import LinearRegression, LogisticRegression

linear_regression = LinearRegression()
logistic_regression = LogisticRegression(random_state=42)

linear_regression.fit(X_train, y_train_linear)
logistic_regression.fit(X_train, y_train_logistic);

from sklearn.metrics import accuracy_score

y_pred_lin = np.sign(linear_regression.predict(X_test))
y_pred_log = logistic_regression.predict(X_test)

accuracy_lin = accuracy_score(np.sign(y_test_linear), y_pred_lin)
accuracy_log = accuracy_score(y_test_logistic, y_pred_log)

print(f"Linear Regression Accuracy: {accuracy_lin*100:,.2f}%")
print(f"Logistic Regression Accuracy: {accuracy_log*100:,.2f}%")

backtest_data = prices.filter(stock)[-len(y_pred_lin):]
backtest_data['lin_reg_signal'] = y_pred_lin
backtest_data['log_reg_signal'] = y_pred_log
backtest_data['lin_reg_signal'] = backtest_data['lin_reg_signal'].shift()
backtest_data['log_reg_signal'] = backtest_data['log_reg_signal'].shift()
models = ['lin_reg', 'log_reg']

transactions = {
    model: {
        'buys': [],
        'sells': [],
    } for model in models
}

def log_transactions(transactions, models, backtest_data):
    for model in models:
        in_position = False
        
        for _, row in backtest_data.iterrows():
            signal = row[f'{model}_signal']

            if signal == 1 and not in_position:
                in_position = True
                transactions[model]['buys'].append(row[stock].values[0])
            
            elif signal == -1 and in_position:
                in_position = False
                transactions[model]['sells'].append(row[stock].values[0])
            
        transactions[model]['profit'] = [(sell - buy) / buy for sell, buy in zip(transactions[model]['sells'], transactions[model]['buys'])]
        transactions[model]['profit'].insert(0, 0)
        transactions[model]['profit'] = (np.cumprod(1 + np.array(transactions[model]['profit'])) - 1)[-1]
    return transactions

transactions = log_transactions(transactions, models, backtest_data)

for model in models:
    print(f'{model}: \t {transactions[model]["profit"]*100:,.2f}%')

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# Create a GridSearchCV object
log_reg_grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5)

# Fit to the data
log_reg_grid.fit(X_train, y_train_logistic)

# Get the best parameters and model
tuned_params = log_reg_grid.best_params_
log_reg_tuned = log_reg_grid.best_estimator_

# Predict on the test set
y_pred_log_tuned = log_reg_tuned.predict(X_test)

# Evaluate accuracy
accuracy_log_tuned = accuracy_score(y_test_logistic, y_pred_log_tuned)

print("Tuned Logistic Regression Accuracy:", accuracy_log_tuned)
print("Best Parameters:", tuned_params)

backtest_data['log_reg_tuned_signal'] = y_pred_log_tuned
backtest_data['log_reg_tuned_signal'] = backtest_data['log_reg_tuned_signal'].shift()

all_models = models + ['log_reg_tuned']

transactions2 = {
    model: {
        'buys': [],
        'sells': [],
    } for model in all_models
}

transactions2 = log_transactions(transactions2, all_models, backtest_data)

for model in all_models:
    print(f'{model}: \t {transactions2[model]["profit"]*100:,.2f}%')