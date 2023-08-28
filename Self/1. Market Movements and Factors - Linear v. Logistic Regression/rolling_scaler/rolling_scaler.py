from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class RollingWindowScaler(BaseEstimator, TransformerMixin):
    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self  # No fitting necessary

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_scaled = X.copy()
            for col in X.columns:
                X_scaled[col] = self._scale_series(X[col])
            return X_scaled
        else:
            raise ValueError("Input should be a Pandas DataFrame")

    def _scale_series(self, series):
        scaled_values = np.empty_like(series)
        for i in range(len(series)):
            start_idx = max(0, i - self.window_size + 1)
            end_idx = i + 1
            window_data = series[start_idx:end_idx]
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std != 0:
                scaled_values[i] = (series[i] - mean) / std
            else:
                scaled_values[i] = 0
        return pd.Series(scaled_values, index=series.index)

# Example usage with Pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', RollingWindowScaler(window_size=10))
])

# Sample data
data = {'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
df = pd.DataFrame(data)

# Apply pipeline
scaled_df = pipeline.fit_transform(df)
print("Scaled DataFrame:")
print(scaled_df)
