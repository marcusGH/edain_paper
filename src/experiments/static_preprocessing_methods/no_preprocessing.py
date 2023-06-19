import yaml
import sklearn
import os
import numpy as np
import src.experiments.static_preprocessing_methods.experiment_setup as setup
from src.lib import experimentation

with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

class IdentityTransform(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X

torch.manual_seed(42)
np.random.seed(42)

history = experimentation.cross_validate_model(
    model=setup.model,
    loss_fn=setup.loss_fn,
    data_loader_kwargs=setup.data_loader_kwargs,
    fit_kwargs=setup.fit_kwargs,
    fill_dict=setup.fill_dict,
    corrupt_func=setup.undo_min_max_corrupt_func,
    preprocess_init_fn=lambda : IdentityTransform(),
    device_ids=None,
)

np.save(os.path.join(cfg['experiment_directory'], 'no-preprocess-history.npy'), history)
