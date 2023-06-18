import yaml
import os
import importlib
import numpy as np
import torch
import torch.nn.functional as F
import sklearn

from src.lib import experimentation
from src.models import basic_grunet
importlib.reload(experimentation)
importlib.reload(basic_grunet)

with open(os.path.join("config.yaml")) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

def identity_corrupt(X, y) -> (np.ndarray, np.ndarray):
    """
    X of shape (num_examples, series_length, num_features)
    """
    return X, y

class IdentityTransform(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        # https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch
        # consider using cupy if too slow with base numpy ^^
        return self

    def transform(self, X):
        return X

data_loader_kwargs = {
        'batch_size' : 1024,
        'shuffle' : True,
        'drop_last' : True,
        }

fill_dict = {
        'nan' : -0.5,
        'pad_categorical' : -2,
        'pad_numeric' : -1.,
        }

model = basic_grunet.GRUNetBasic(188, 128, 2, 4).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.01)
#early_stopper = experimentation.EarlyStopper(patience=5, min_delta=0.0)
loss_fn = F.binary_cross_entropy

train_loader, val_loader = experimentation.load_numpy_data(
        os.path.join(cfg['dataset_directory'], "derived", "processed-splits"),
        val_idx=[0, 1],
        fill_dict=fill_dict,
        preprocess_obj=IdentityTransform(),
        **data_loader_kwargs)



history = experimentation.fit_model(model, loss_fn, train_loader, val_loader, optimizer, scheduler,
        num_epochs = 1000, verbose = False, early_stopper = None)
np.save(os.path.join(cfg['checkpoint_directory'], 'no_early_stopper_history'), history)
