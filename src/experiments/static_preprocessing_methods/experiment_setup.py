import yaml
import os
import importlib
import numpy as np
import torch
import torch.nn.functional as F
import sklearn

from src.lib import experimentation
from src.models import basic_grunet

# undo the min-max preprocessing
def undo_min_max_corrupt_func(X, y):
    """
    X of shape (num_examples, series_length, num_features)
    In this undo, we assume scale same for each feature, over temporal scale
    """
    # to ensure we get the same mins and scales every time
    np.random.seed(42)
    # randomize both the starting point and the feature scales
    mins = np.random.uniform(-1E5, 1E5, size=X.shape[2])[np.newaxis, None]
    scales = 10 ** np.random.uniform(-5, 5, size=X.shape[2])[np.newaxis, None]

    X_corrupt = X * scales + mins
    return X_corrupt, y

with open(os.path.join("config.yaml")) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

data_loader_kwargs = {
    'batch_size' : 1024,
    'shuffle' : False,
    'drop_last' : False,
}

fit_kwargs = {
    'train_split_data_dir' : os.path.join(cfg['dataset_directory'], "derived", "processed-splits"),
    'num_epochs' : 20,
    'early_stopper_patience' : 3,
    'early_stopper_min_delta' : 0.0025,
    'optimizer_init' : lambda x: torch.optim.Adam(x, lr = 0.001),
    'scheduler_init' : lambda x: torch.optim.lr_scheduler.MultiStepLR(x, milestones=[4, 7], gamma=0.1),
    'verbose' : False,
}

fill_dict = {
    'nan' : -0.5,
    'pad_categorical' : -2,
    'pad_numeric' : -1.,
}

torch.manual_seed(42)
np.random.seed(42)

model = basic_grunet.GRUNetBasic(188, 128, 2, 4)
loss_fn = F.binary_cross_entropy
