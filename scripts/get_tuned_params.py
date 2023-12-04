import re
import os
from src.lib import plotting

cfg = plotting.get_config()


def get_best_hist_cfg(exp_name):
    files = [f for f in os.listdir(cfg['experiment_directory']) if re.match(re.escape(exp_name) + r'-\d+.*', f)]
    best_vloss = float("inf")
    ret_hist = None
    for f in files:
        # remove the .npy ending
        hist = plotting.load_hist(f[:-4])
        if hist['val_loss'][0][-1] < best_vloss:
            best_vloss = hist['val_loss'][0][-1]
            ret_hist = hist
    return ret_hist

print("Global-aware")
print(get_best_hist_cfg('hpc-tuning')['experiment_config']['edain_bijector_fit'])
print("Local-aware")
print(get_best_hist_cfg('hpc-ba-tuning')['experiment_config']['edain_bijector_fit'])

