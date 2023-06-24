import yaml
import sklearn
import os
import numpy as np
import torch
import src.experiments.dain.dain_experiment_setup as setup
from src.experiments.static_preprocessing_methods import winsorization
from src.lib import experimentation
from sklearn import preprocessing

with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

class MinMaxTimeSeries(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self, a=0, b=1, time_series_length=13):
        self.T = time_series_length
        self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(a, b))

    def fit(self, X, y = None):
        # merge the dimensions and time axis
        X = X.reshape((X.shape[0], -1))
        self.min_max_scaler.fit(X, y)
        return self

    def transform(self, X):
        X = X.reshape((X.shape[0], -1))
        # scale all the features
        X = self.min_max_scaler.transform(X)
        return X.reshape((X.shape[0], self.T, -1))

torch.manual_seed(42)
np.random.seed(42)

history = experimentation.cross_validate_model(
    model=setup.model,
    loss_fn=setup.loss_fn,
    data_loader_kwargs=setup.data_loader_kwargs,
    fit_kwargs=setup.fit_kwargs,
    fill_dict=setup.fill_dict,
    corrupt_func=setup.undo_min_max_corrupt_func,
    preprocess_init_fn=lambda : MinMaxTimeSeries(),
    device_ids=[6],
)

np.save(os.path.join(cfg['experiment_directory'], 'dain5-min-max-scaling-history-50-epochs.npy'), history)
