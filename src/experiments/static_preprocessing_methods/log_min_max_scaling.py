import yaml
import sklearn
import os
import numpy as np
import torch
from src.experiments.static_preprocessing_methods import winsorization
from src.lib import experimentation
from sklearn import preprocessing

with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

class LogMinMaxTimeSeries(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self, a=0, b=1, alpha=1.96, time_series_length=13):
        self.T = time_series_length
        self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(a, b))
        self.alpha = alpha

    def fit(self, X, y = None):
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

if __name__ == "__main__":
    import src.experiments.static_preprocessing_methods.experiment_setup as setup

    torch.manual_seed(42)
    np.random.seed(42)

    history = experimentation.cross_validate_model(
        model=setup.model,
        loss_fn=setup.loss_fn,
        data_loader_kwargs=setup.data_loader_kwargs,
        fit_kwargs=setup.fit_kwargs,
        fill_dict=setup.fill_dict,
        corrupt_func=setup.undo_min_max_corrupt_func,
        # preprocess_init_fn=lambda : MinMaxTimeSeries(),
        preprocess_init_fn=lambda : LogMinMaxTimeSeries(a=0, b=1, alpha=0.025),
        device_ids=[2],
    )

    np.save(os.path.join(cfg['experiment_directory'], 'log-min-max-scaling-history-50-epochs.npy'), history)
