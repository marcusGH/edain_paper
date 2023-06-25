import yaml
import os
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import importlib.util
import sys

from src.lib import experimentation
from src.models.dain_grunet import DainGRUNet
import src.experiments.static_preprocessing_methods.experiment_setup as spm

def undo_min_max_corrupt_func(X, y):
    """
    X of shape (num_examples, series_length, num_features)
    In this undo, we assume scale same for each feature, over temporal scale
    """
    # to ensure we get the same mins and scales every time
    np.random.seed(42)
    # randomize both the starting point and the feature scales
    mins = np.random.uniform(-1E4, 1E4, size=X.shape[2])[np.newaxis, None]
    # don't set the smallest scale too tiny, otherwise can lose information due to float 32 bit
    scales = 10 ** np.random.uniform(-1, 5, size=X.shape[2])[np.newaxis, None]

    X_corrupt = X * scales + mins
    return X_corrupt, y

with open(os.path.join("config.yaml")) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

data_loader_kwargs = {
    'batch_size' : 1024,
    'shuffle' : False,
    'drop_last' : False,
}

def optimizer_init(mod, lr=1e-3):
    return torch.optim.Adam([
        {'params': model.gru.parameters()},
        {'params': model.emb_layers.parameters()},
        {'params': model.feed_forward.parameters()},
        {'params': model.dain.mean_layer.parameters(), 'lr': lr * model.dain.mean_lr},
        {'params': model.dain.scaling_layer.parameters(), 'lr': lr * model.dain.scale_lr},
        {'params': model.dain.gating_layer.parameters(), 'lr': lr * model.dain.gate_lr},
    ], lr=lr)


fit_kwargs = {
        'train_split_data_dir' : os.path.join(cfg['dataset_directory'], "derived", "processed-splits"),
        'num_epochs' : 50,
        'early_stopper_patience' : 5,
        'early_stopper_min_delta' : 0.0,
        'optimizer_init' : optimizer_init,
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
# optimal learning rates for RNN according to paper
# mean_lr, std_lr, scale_lr = 1e-02, 1e-8, 10 # Used for experiment 3.
# mean_lr, std_lr, scale_lr = 0.01, 0.01, 10 # Used for experiment 4.
mean_lr, std_lr, scale_lr = 1, 1, 1 # Used for experiment 5.
model = DainGRUNet(188, 128, 2, 4, mode='adaptive_scale', mean_lr=mean_lr, scale_lr=std_lr, gate_lr=scale_lr)
loss_fn = F.binary_cross_entropy
