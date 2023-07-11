import yaml
import sklearn
import os
import numpy as np
import torch
import src.experiments.dain.dain_experiment_setup as setup
from src.experiments.static_preprocessing_methods import winsorization
from src.experiments.static_preprocessing_methods.standard_scaling import StandardScalerTimeSeries
from src.models.adaptive_grunet import AdaptiveGRUNet
import src.preprocessing.adaptive_transformations as at
from src.lib import experimentation
from sklearn import preprocessing

with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

def dain_full_optimizer_init(model, lr=1e-3):
    return torch.optim.Adam([
        {'params': model.gru.parameters()},
        {'params': model.emb_layers.parameters()},
        {'params': model.feed_forward.parameters()},
        {'params': model.preprocess.winsorization_params(), 'lr': lr * 1e-1},
        {'params': model.preprocess.power_transform_params(), 'lr': lr * 1e-6},
        {'params': model.preprocess.scaling_params(), 'lr': lr * 1e-1},
        {'params': model.preprocess.shift_params(), 'lr': lr * 1e-1},
    ], lr=lr)

torch.manual_seed(42) ; np.random.seed(42)
dain_model = AdaptiveGRUNet(lambda D, T: at.FullDAIN_Layer(input_dim=D, time_series_length=T, adaptive_scale=False, adaptive_power_transform=False, adaptive_shift=False, adaptive_winsorization=True, dev=torch.device('cuda', 1)), 188, 128, 2, 4)
fit_kwargs = setup.fit_kwargs
fit_kwargs['optimizer_init'] = dain_full_optimizer_init

torch.manual_seed(42) ; np.random.seed(42)
history = experimentation.cross_validate_model(
    model=dain_model,
    loss_fn=setup.loss_fn,
    data_loader_kwargs=setup.data_loader_kwargs,
    fit_kwargs=fit_kwargs,
    fill_dict=setup.fill_dict,
    corrupt_func=setup.undo_min_max_corrupt_func,
    preprocess_init_fn=lambda : StandardScalerTimeSeries(13),
    device_ids=[1],
)

np.save(os.path.join(cfg['experiment_directory'], 'full-dain-standard-scaling-only-adaptive-winsorization-history-50-epochs.npy'), history)
