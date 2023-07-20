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
        # preprocess_init_fn=lambda : TanhStandardScalerTimeSeries(),
        preprocess_init_fn=lambda : winsorization.WinsorizeDecorator(TanhStandardScalerTimeSeries, alpha=0.05),
        device_ids=[3],
    )

    np.save(os.path.join(cfg['experiment_directory'], 'tanh-standard-scaling-history-50-epochs-winsorized.npy'), history)
