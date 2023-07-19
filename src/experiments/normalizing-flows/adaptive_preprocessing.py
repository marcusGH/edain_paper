import yaml
import sklearn
import os
import numpy as np
import torch
from src.lib import experimentation
from src.experiments.static_preprocessing_methods.min_max_scaling import MinMaxTimeSeries
from src.preprocessing.normalizing_flows import AdaptivePreprocessingLayerTimeSeries
from sklearn import preprocessing

with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

class MinMaxAdaptivePreprocessingLayerTimeSeries(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """
    Wrapper that does min max transform before feeding normalized data into the adapter KL optimizer
    """
    def __init__(self, time_series_length=13, input_dim=177):
        bijector_kwargs = {
            'init_sigma' : 0.3,
            'eps' : 1e-6,
            'adaptive_shift' : True,
            'adaptive_scale' : True,
            'adaptive_outlier_removal' : True,
            'adaptive_power_transform' : False,
            'outlier_removal_mode' : 'exp',
        }
        bijector_fit_kwargs = {
            'batch_size' : 1024,
            'device' : torch.device('cuda', 4),
            'milestones' : [3, 7],
            'num_epochs' : 20,
            # learning rates
            'base_lr' : 1e-3,
            'scale_lr' : 10,
            'shift_lr' : 10,
            'outlier_lr' : 1,
            'power_lr' : 1e-7,
        }
        self.min_max = MinMaxTimeSeries(time_series_length=time_series_length)
        self.adaptive_layer = AdaptivePreprocessingLayerTimeSeries(time_series_length, input_dim, bijector_kwargs, bijector_fit_kwargs)

    def fit(self, X, y=None):
        X = self.min_max.fit_transform(X, y)
        self.adaptive_layer.fit(X, y)
        return self

    def transform(self, X):
        X = self.min_max.transform(X)
        X = self.adaptive_layer.transform(X)
        return X

if __name__ == "__main__":
    import src.experiments.static_preprocessing_methods.experiment_setup as setup

    torch.manual_seed(101)
    np.random.seed(101)

    history = experimentation.cross_validate_model(
        model=setup.model,
        loss_fn=setup.loss_fn,
        data_loader_kwargs=setup.data_loader_kwargs,
        fit_kwargs=setup.fit_kwargs,
        fill_dict=setup.fill_dict,
        corrupt_func=setup.undo_min_max_corrupt_func,
        preprocess_init_fn=lambda : MinMaxAdaptivePreprocessingLayerTimeSeries(),
        # Use same device for both training bijector and RNN (although we don't have to)
        device_ids=[4],
    )

    np.save(os.path.join(cfg['experiment_directory'], 'adaptive-preprocessing-kl-6-inverse.npy'), history)
