#!/usr/bin/env python
from src.lib.plotting import load_hist
import numpy as np
import pandas as pd
import torch

DEV = torch.device('cuda', 0)

methods = ["raw", "z_score", "inverse_CDF", "bin", "dain", "edain_local", "edain_global", "edain-kl"]

hists = [load_hist(f"synth_data_performance_{lab}") for lab in methods]

hists.append(load_hist("mcCarter-synth-0.1"))
hists.append(load_hist("mcCarter-synth-1"))
hists.append(load_hist("mcCarter-synth-10"))
hists.append(load_hist("mcCarter-synth-100"))
methods.extend(['McCarter ($\\alpha=0.1$)', 'McCarter ($\\alpha=1$)', 'McCarter ($\\alpha=10$)', 'McCarter ($\\alpha=100$)'])

res_data = {
    "method" : [],
    "val_loss" : [],
    # "val_amex_metric" : [],
    "val_accs" : [],
    # "num_epochs" : [],
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
