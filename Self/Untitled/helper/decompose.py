from sklearn.decomposition import PCA
import pandas as pd

class PCATransform:

    def __init__(self, random_state=None, n_components=None):
        self.pca = PCA(
            n_components=n_components, 
            random_state=random_state
        )
        self.n_components = n_components
        self.random_state = random_state
    
    def fit(self, data):
        if self.n_components is None:
            self.pca = PCA(
                n_components=len(data.columns),
                random_state=self.random_state
            )
        t = self.pca.fit_transform(data)
        t = pd.DataFrame(t)
        t.rename(columns={
            col: f'PC{col+1}' for col in t.columns},
            inplace=True
        )
        t.index = data.index
        self.transformed = t
        self._get_evr()

    def _get_evr(self):
        evr = self.pca.explained_variance_ratio_
        evr = pd.DataFrame(evr)
        evr.rename(columns={0: 'EVR'}, inplace=True)
        evr.index = self.transformed.columns
        self.explained_variance_ratio = evr
