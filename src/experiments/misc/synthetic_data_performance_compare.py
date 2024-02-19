#!/usr/bin/env python

# imports
from datetime import datetime
from scipy import stats
from scipy.integrate import quad, cumulative_trapezoid
from src.lib import experimentation
from src.lib.synthetic_data import SyntheticData
from src.models.adaptive_grunet import AdaptiveGRUNet
from src.models.basic_grunet import GRUNetBasic
from src.preprocessing.adaptive_transformations import DAIN_Layer, BiN_Layer
from src.preprocessing.normalizing_flows import EDAIN_Layer, EDAINScalerTimeSeries, EDAINScalerTimeSeriesDecorator
from src.preprocessing.static_transformations import StandardScalerTimeSeries, McCarterTimeSeries, BaselineTransform
from tqdm.auto import tqdm

import sys
import copy
import cudf
import cupy
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

with open(os.path.join("config.yaml")) as f:
    main_cfg = yaml.load(f, Loader=yaml.FullLoader)

with open(os.path.join("src", "experiments", "configs", "experiment-config-alpha.yaml")) as f:
    amex_cfg = yaml.load(f, Loader=yaml.FullLoader)

with open(os.path.join("src", "experiments", "configs", "experiment-config-beta.yaml")) as f:
    lob_cfg = yaml.load(f, Loader=yaml.FullLoader)

DEV = torch.device('cuda', 0)
# DEV = torch.device('cpu')

def evaluate_model(datasets, preprocess_init_fn=None, model_init_fn=None, input_dim=3, return_model=False):
    tlosses = []
    vlosses = []
    tmetrics = []
    vmetrics = []
    accs = []
    num_epochs = []

    for X, y in tqdm(datasets):
        N = X.shape[0]

        if model_init_fn is None:
            model = GRUNetBasic(input_dim, 32, 2, 2, 0)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        else:
            model = model_init_fn()
            optimizer = torch.optim.Adam([
                    {'params' : model.preprocess.parameters(), 'lr' : 1e-2},
                    {'params' : model.gru.parameters(), 'lr' : 1e-3},
                    {'params' : model.feed_forward.parameters(), 'lr' : 1e-3}
                ], lr = 0.001)
        loss_fn = F.binary_cross_entropy

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 7], gamma=0.1)
        early_stopper = experimentation.EarlyStopper(patience=5)

        # train-val split
        X_train, y_train = X[:int(N * 0.8)], y[:int(N * 0.8)]
        X_val, y_val = X[int(N * 0.8):], y[int(N * 0.8):]

        # preprocess the dataset if provided with a method
        if preprocess_init_fn is not None:
            print(f"Before preprocess: {X_train.shape}, y_train: {y_train.shape}")
            preprocess = preprocess_init_fn()
            X_train = preprocess.fit_transform(X_train, y_train)
            X_val = preprocess.transform(X_val)
            print("Preprocessing finished!")

        print(f"Starting training with X_train: {X_train.shape}, y_train: {y_train.shape}")

        train_loader = torch.utils.data.DataLoader(
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(X_train).type(torch.float32),
                    torch.from_numpy(y_train).type(torch.float32)),
                batch_size=128, shuffle = True)
        val_loader = torch.utils.data.DataLoader(
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(X_val).type(torch.float32),
                    torch.from_numpy(y_val).type(torch.float32)),
                batch_size=128, shuffle = True)

        history = experimentation.fit_model(model, loss_fn, train_loader, val_loader, optimizer, num_epochs=NUM_EPOCHS, early_stopper=early_stopper, scheduler=scheduler, verbose=False, device_ids=DEV)

        if return_model and model_init_fn is not None:
            if preprocess_init_fn is not None:
                return model, preprocess
            else:
                return model
        elif return_model and preprocess_init_fn is not None:
            return preprocess
        elif return_model:
            raise ValueError("Invalid combination of arguments")

        # compute the classification binary accuracy
        val_preds = []
        val_labs = []
        for x_, y_ in val_loader:
            val_preds.append(model(x_.to(DEV)).detach().cpu().numpy())
            val_labs.append(y_.cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_labs = np.concatenate(val_labs)

        vlosses.append(history['val_loss'][-1])
        tlosses.append(history['train_loss'][-1])
        vmetrics.append(history['val_amex_metric'][-1])
        tmetrics.append(history['train_amex_metric'][-1])
        accs.append(np.mean(np.where(val_preds > 0.5, 1.0, 0.0) == val_labs))
        num_epochs.append(len(history['train_loss']))

    return {
        'val_loss' : np.array(vlosses),
        'val_amex_metric' : np.array(vmetrics),
        'train_loss' : np.array(tlosses),
        'train_amex_metric' : np.array(tmetrics),
        'val_accs' : np.array(accs),
        'num_epochs' : np.array(num_epochs),
    }

def cdf_invert(f, x, A=-50, B=50, n=40000):
    # cache the integral computation
    if f not in cdf_invert.cache:
        xs = np.linspace(A, B, n)
        Fs = cumulative_trapezoid(f(xs), xs, initial=0)
        Fs /= np.max(Fs)
        cdf_invert.cache[f] = Fs
    assert np.all(x > A) and np.all(x < B)
    # find the appropriate index by bucketing
    idx = np.array((x - A) * n / (B - A), dtype=np.int32)
    return cdf_invert.cache[f][idx]
cdf_invert.cache = {}

class UndoCorruptionTimeSeries(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self, pdfs, time_series_length = 10, epsilon=1e-4):
        self.epsilon = epsilon
        self.T = time_series_length
        self.pdfs = pdfs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = copy.deepcopy(X)
        assert X.shape[2] == len(self.pdfs)
        for t in range(self.T):
            for j, f in enumerate(self.pdfs):
                # inverse normal CDF * original CDF
                X[:, t, j] = stats.norm.ppf(cdf_invert(f, X[:, t, j]) * (1 - self.epsilon) + self.epsilon / 2)
        return X

D = 3
T = 10
# lower bound, upper bound, and unormalized PDF
bounds = [(-8, 10), (-30, 30), (-1, 7)]
f1 = lambda x: 10 * stats.norm.cdf(10 * (x+4)) * stats.norm.pdf(x+4) + 0.1 * np.where(x > 8, np.exp(x - 8), 0) * np.where(x < 9.5, np.exp(9.5 - x), 0)
f2 = lambda x: np.where(x > np.pi, 20 * stats.norm.pdf(x-20), np.exp(x / 6) * (10 * np.sin(x) + 10))
f3 = lambda x: 2 * stats.norm.cdf(-4 * (x-4)) * stats.norm.pdf(x - 4)
# both of the two time-series will be q=3 and q=2, respecitvely
thetas = np.array([
    [-1., 0.5, -0.2, 0.8],
    [-1., 0.3, 0.9, 0.0],
    [-1., 0.8, 0.3, -0.9],
])
CROSS_VAR_SIGMA = 1.4
RESPONSE_NOISE_SIGMA = 0.5
RESPONSE_BETA_SIGMA = 2.0
RANDOM_STATE = 42
NUM_DATASETS = 100
NUM_EPOCHS = 30
NUM_SAMPLES = 50000

synth_data = SyntheticData(
    dim_size=D,
    time_series_length=T,
    pdfs = [f1, f2, f3],
    ar_q = thetas.shape[1] - 1,
    ar_thetas=thetas,
    pdf_bounds=bounds,
    cross_variables_cor_init_sigma=CROSS_VAR_SIGMA,
    response_noise_sigma=RESPONSE_NOISE_SIGMA,
    response_beta_sigma=RESPONSE_BETA_SIGMA,
    random_state=RANDOM_STATE,
)

# generate the datasets
datasets = []
for i in tqdm(range(NUM_DATASETS)):
    X_raw, y_raw = synth_data.generate_data(n=NUM_SAMPLES, return_uniform=False, random_state=i)
    datasets.append((X_raw, y_raw))


# evaluate the statistical baselines
print("Evaluating baseline 111")
hist = evaluate_model(datasets, preprocess_init_fn=lambda : BaselineTransform(10, True, True, True), return_model=False)
np.save(os.path.join(main_cfg['experiment_directory'], "synth_data_performance_baseline-111.npy"), hist)

print("Evaluating baseline 010")
hist = evaluate_model(datasets, preprocess_init_fn=lambda : BaselineTransform(10, False, True, False), return_model=False)
np.save(os.path.join(main_cfg['experiment_directory'], "synth_data_performance_baseline-010.npy"), hist)

print("Evaluating baseline 011")
hist = evaluate_model(datasets, preprocess_init_fn=lambda : BaselineTransform(10, False, True, True), return_model=False)
np.save(os.path.join(main_cfg['experiment_directory'], "synth_data_performance_baseline-011.npy"), hist)

print("Evaluating baseline 110")
hist = evaluate_model(datasets, preprocess_init_fn=lambda : BaselineTransform(10, True, True, False), return_model=False)
np.save(os.path.join(main_cfg['experiment_directory'], "synth_data_performance_baseline-110.npy"), hist)

sys.exit()

print("Evaluating McCarter 0.1...")
hist = evaluate_model(datasets, preprocess_init_fn=lambda : McCarterTimeSeries(10, alpha=0.1), return_model=False)
np.save(os.path.join(main_cfg['experiment_directory'], "mcCarter-synth-0.1.npy"), hist)

print("Evaluating McCarter 1...")
hist = evaluate_model(datasets, preprocess_init_fn=lambda : McCarterTimeSeries(10, alpha=1), return_model=False)
np.save(os.path.join(main_cfg['experiment_directory'], "mcCarter-synth-1.npy"), hist)

print("Evaluating McCarter 10...")
hist = evaluate_model(datasets, preprocess_init_fn=lambda : McCarterTimeSeries(10, alpha=10), return_model=False)
np.save(os.path.join(main_cfg['experiment_directory'], "mcCarter-synth-10.npy"), hist)

print("Evaluating McCarter 100...")
hist = evaluate_model(datasets, preprocess_init_fn=lambda : McCarterTimeSeries(10, alpha=100), return_model=False)
np.save(os.path.join(main_cfg['experiment_directory'], "mcCarter-synth-100.npy"), hist)

sys.exit()

print("Evaluating raw data...")
hist = evaluate_model(datasets, preprocess_init_fn=None, return_model=False)
np.save(os.path.join(main_cfg['experiment_directory'], "synth_data_performance_raw.npy"), hist)

print("Evaluating standard scaling...")
hist = evaluate_model(datasets, preprocess_init_fn=lambda : StandardScalerTimeSeries(10), return_model=False)
np.save(os.path.join(main_cfg['experiment_directory'], "synth_data_performance_z_score.npy"), hist)

print("Evaluating inverse CDF...")
hist = evaluate_model(datasets, preprocess_init_fn=lambda : UndoCorruptionTimeSeries([f1, f2, f3], 10))
np.save(os.path.join(main_cfg['experiment_directory'], "synth_data_performance_inverse_CDF.npy"), hist)

print("Evaluating BIN...")
bin_init_fn = lambda : AdaptiveGRUNet(
    BiN_Layer(input_shape=(3, 10)),
    3, 32, 2, 2, 0, 10, dim_first=True)
hist = evaluate_model(datasets, model_init_fn=bin_init_fn)
np.save(os.path.join(main_cfg['experiment_directory'], "synth_data_performance_bin.npy"), hist)

print("Evaluating DAIN...")
dain_init_fn = lambda : AdaptiveGRUNet(
    DAIN_Layer(mode='adaptive_scale', input_dim=3),
    3, 32, 2, 2, 0, 10, dim_first=True)
hist = evaluate_model(datasets, model_init_fn=dain_init_fn)
np.save(os.path.join(main_cfg['experiment_directory'], "synth_data_performance_dain.npy"), hist)

print("Evaluating EDAIN (local)...")
edain_local_init_fn = lambda : AdaptiveGRUNet(
    EDAIN_Layer(
        input_dim=3,
        invert_bijector=False,
        outlier_removal_residual_connection=True,
        batch_aware=True,
        init_sigma=0.000001,
        outlier_removal_mode='exp',
    ), 3, 32, 2, 2, 0, 10, dim_first=False)
hist = evaluate_model(datasets, model_init_fn=edain_local_init_fn)
np.save(os.path.join(main_cfg['experiment_directory'], "synth_data_performance_edain_local.npy"), hist)

print("Evaluating EDAIN (global)")
edain_global_init_fn = lambda : AdaptiveGRUNet(
    EDAIN_Layer(
        input_dim=3,
        invert_bijector=False,
        outlier_removal_residual_connection=True,
        batch_aware=False,
        init_sigma=0.000001,
        outlier_removal_mode='exp',
    ), 3, 32, 2, 2, 0, 10, dim_first=False)
hist = evaluate_model(datasets, preprocess_init_fn=lambda : StandardScalerTimeSeries(10), model_init_fn=edain_global_init_fn)
np.save(os.path.join(main_cfg['experiment_directory'], "synth_data_performance_edain_global.npy"), hist)

print("Evaluating EDAIN-KL")
hist = evaluate_model(datasets, preprocess_init_fn=lambda : EDAINScalerTimeSeriesDecorator(StandardScalerTimeSeries(10), 10, 3,
        bijector_kwargs={
            'init_sigma' : 0.0001,
            'eps' : 0.00001,
            'batch_aware' : False,
            'outlier_removal_mode' : 'exp',
        },
        bijector_fit_kwargs={
            'device' : DEV,
            'scale_lr' : 10,
            'shift_lr' : 10,
            'outlier_lr' : 10.0,
            'power_lr' : 10.0,
            'batch_size' : 1024,
            'milestones' : [4, 8],
            'num_epochs' : 20,
            'num_fits' : 1,
            'base_lr': 1e-3,
        }))
np.save(os.path.join(main_cfg['experiment_directory'], "synth_data_performance_edain-kl.npy"), hist)
