#!/usr/bin/env python
from datetime import datetime
from scipy import stats
from scipy.integrate import quad, cumulative_trapezoid
from src.lib import experimentation
from src.lib.plotting import load_hist
from src.lib.synthetic_data import SyntheticData
from src.models.adaptive_grunet import AdaptiveGRUNet
from src.models.basic_grunet import GRUNetBasic
from src.preprocessing.adaptive_transformations import DAIN_Layer, BiN_Layer
from src.preprocessing.normalizing_flows import EDAIN_Layer, EDAINScalerTimeSeries, EDAINScalerTimeSeriesDecorator
from src.preprocessing.static_transformations import StandardScalerTimeSeries
from tqdm.auto import tqdm

import copy
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

DEV = torch.device('cuda', 0)

methods = ["raw", "z_score", "inverse_CDF", "bin", "dain", "edain_local", "edain_global", "edain-kl"]

hists = [load_hist(f"synth_data_performance_{lab}") for lab in methods]

res_data = {
    "method" : [],
    "val_loss" : [],
    "val_amex_metric" : [],
    "val_accs" : [],
    "num_epochs" : [],
}
for h, lab in zip(hists, methods):
    for k in res_data.keys():
        if "improvement" in k:
            continue
        elif k == "method":
            res_data[k].append(lab)
        else:
            m = np.mean(h[k])
            s = np.std(h[k])
            res_data[k].append(f"${m:.4f} \pm {1.96 * s:.4f}$")

df = pd.DataFrame(res_data)
print(df.to_latex(index=False))
