import numpy as np
import sklearn

class WinsorizeDecorator(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """
    Before fitting provided transformer winsorizes the data to be within
    the [alpha/2, 1-alpha/2] quantiles of the fitted data.
    """
    def __init__(self, TransformerClass, alpha=0.05, time_series_length=13, **kwargs):
        self.transformer = TransformerClass(time_series_length=time_series_length, **kwargs)
        self.T = time_series_length
        self.alpha = alpha

    def fit(self, X, y = None):
        # merge the dimensions and time axis
        X = X.reshape((X.shape[0], -1))
        # save the quantiles for each of the T * D variables
        self.lower = np.quantile(X.reshape((X.shape[0], -1)), q=self.alpha/2, axis=0)
        self.upper = np.quantile(X.reshape((X.shape[0], -1)), q=1-self.alpha/2, axis=0)
        # transform the data
        X = np.clip(X, self.lower, self.upper)
        # then fit the provided transformer on this winsorized data
        self.transformer = self.transformer.fit(X.reshape((X.shape[0], self.T, -1)), y)
        return self

    def transform(self, X):
        X = X.reshape((X.shape[0], -1))
        # clip all the features based on the learned percentiles
        X = np.clip(X, self.lower, self.upper)
        # apply the provided transformer
        return self.transformer.transform(X.reshape((X.shape[0], self.T, -1)))
