import numpy as np
import sklearn
from sklearn import preprocessing

def identity_corrupt(X, y) -> (np.ndarray, np.ndarray):
    """
    X of shape (num_examples, series_length, num_features)
    """
    return X, y

class IdentityTransform(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X

class StandardScalerTimeSeries(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self, time_series_length : int = 13):
        self.ss = preprocessing.StandardScaler()
        self.T = time_series_length

    def fit(self, X, y = None):
        assert X.shape[1] == self.T
        X = X.reshape((X.shape[0], -1))
        self.ss.fit(X, y)
        return self

    def transform(self, X):
        assert X.shape[1] == self.T
        X = X.reshape((X.shape[0], -1))
        X = self.ss.transform(X)
        X = X.reshape((X.shape[0], self.T, -1))
        return X


class LogMinMaxTimeSeries(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self, time_series_length=13, a=0, b=1, alpha=1.96):
        self.T = time_series_length
        self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(a, b))
        self.alpha = alpha

    def fit(self, X, y = None):
        assert X.shape[1] == self.T
        # merge the dimensions and time axis
        X = X.reshape((X.shape[0], -1))

        # find smallest and largest, as we will base our shift on that
        lower = np.min(X, axis=0)
        upper = np.max(X, axis=0)
        # we clip values to be this at the lowest during transform to avoid negatives
        self.lower_clip = lower - (upper - lower) * self.alpha
        # shift all elements up by this value to ensure they are positive
        # Also add one for variables with very small scale, othw might get 0
        self.shift = -lower + (upper - lower) * self.alpha

        X = np.log1p(X + self.shift)
        self.min_max_scaler.fit(X, y)
        return self

    def transform(self, X):
        assert X.shape[1] == self.T
        X = X.reshape((X.shape[0], -1))
        # scale all the features after shifting
        X = np.clip(X, a_min=self.lower_clip, a_max=None) + self.shift
        if np.any(X < 0):
            print(f"Negative values encountered after shift operation.")
            for i in range(X.shape[1]):
                temp = X[X[:, i] <= 0.0, i]
                if temp.shape[0] > 0:
                    print(f"Offending index {i}: {(np.min(X[:, i]) - self.shift[i]):.4f}, {(np.max(X[:, i]) - self.shift[i]):.4f} and shift={(self.shift[i]):.4f}")

            raise ValueError("Negative values passed to log. Aborting!")
        X = self.min_max_scaler.transform(np.log1p(X))
        return X.reshape((X.shape[0], self.T, -1))


class LogStandardScalerTimeSeries(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self, time_series_length=13, alpha=1.96):
        self.T = time_series_length
        self.standard_scaler = preprocessing.StandardScaler()
        self.alpha = alpha

    def fit(self, X, y = None):
        assert X.shape[1] == self.T
        # merge the dimensions and time axis
        X = X.reshape((X.shape[0], -1))

        # find smallest and largest, as we will base our shift on that
        lower = np.min(X, axis=0)
        upper = np.max(X, axis=0)
        # we clip values to be this at the lowest during transform to avoid negatives
        self.lower_clip = lower - (upper - lower) * self.alpha
        # shift all elements up by this value to ensure they are positive
        # Also add one for variables with very small scale, othw might get 0
        self.shift = -lower + (upper - lower) * self.alpha

        # smooth by adding 1
        X = np.log1p(X + self.shift)
        self.standard_scaler.fit(X, y)
        return self

    def transform(self, X):
        assert X.shape[1] == self.T
        X = X.reshape((X.shape[0], -1))
        # scale all the features after shifting, and clip to be as low as observed during training
        X = np.clip(X, a_min=self.lower_clip, a_max=None) + self.shift
        if np.any(X < 0):
            print(f"Negative values encountered after shift operation.")
            for i in range(X.shape[1]):
                temp = X[X[:, i] < 0.0, i]
                if temp.shape[0] > 0:
                    print(f"Offending index {i}: {(np.min(X[:, i]) - self.shift[i]):.4f}, {(np.max(X[:, i]) - self.shift[i]):.4f} and shift={(self.shift[i]):.4f}")

            raise ValueError("Negative values passed to log. Aborting!")
        X = self.standard_scaler.transform(np.log1p(X))
        return X.reshape((X.shape[0], self.T, -1))


class MinMaxTimeSeries(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self, time_series_length=13, a=0, b=1):
        self.T = time_series_length
        self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(a, b))

    def fit(self, X, y = None):
        assert X.shape[1] == self.T
        # merge the dimensions and time axis
        X = X.reshape((X.shape[0], -1))
        self.min_max_scaler.fit(X, y)
        return self

    def transform(self, X):
        assert X.shape[1] == self.T
        X = X.reshape((X.shape[0], -1))
        # scale all the features
        X = self.min_max_scaler.transform(X)
        return X.reshape((X.shape[0], self.T, -1))


class TanhStandardScalerTimeSeries(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self, time_series_length : int = 13):
        self.ss = preprocessing.StandardScaler()
        self.T = time_series_length

    def fit(self, X, y = None):
        assert X.shape[1] == self.T
        X = X.reshape((X.shape[0], -1))
        self.ss.fit(X, y)
        return self

    def transform(self, X):
        assert X.shape[1] == self.T
        X = X.reshape((X.shape[0], -1))
        X = self.ss.transform(X)
        X = X.reshape((X.shape[0], self.T, -1))
        return np.tanh(X)


class WinsorizeDecorator(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """
    Before fitting provided transformer winsorizes the data to be within
    the [alpha/2, 1-alpha/2] quantiles of the fitted data.
    """
    def __init__(self, scaler, alpha=0.05, time_series_length=13):
        self.transformer = scaler
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


class IgnoreTimeDecorator(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """
    Learns the transformation only on the dimension axis, ignore time index
    """
    def __init__(self, scaler, time_series_length=13):
        self.transformer = scaler
        self.T = time_series_length
        self.D = None

    def fit(self, X, y = None):
        assert X.shape[1] == self.T
        self.D = X.shape[2]
        # merge the num_samples and time axis, then add a dummy time axis
        X = X.reshape((-1, 1, self.D))
        # then fit the provided transformer on this data
        self.transformer = self.transformer.fit(X, y)
        return self

    def transform(self, X):
        assert X.shape[1] == self.T
        X = X.reshape((-1, 1, self.D))
        # apply the provided transformer
        X = self.transformer.transform(X)
        # unflatten and return
        return X.reshape((-1, self.T, self.D))
