import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

if os.path.isfile("config.yaml"):
    with open("config.yaml") as f:
        _cfg = yaml.load(f, Loader=yaml.FullLoader)
elif os.path.isfile(os.path.join("..", "config.yaml")):
    with open(os.path.join("..", "config.yaml")) as f:
        _cfg = yaml.load(f, Loader=yaml.FullLoader)
else:
    raise FileNotFoundError("Unable to locate configuration file: config.yaml")

def get_average(history, key):
    """
    Given a historu object and metric key, returns a list
    of the average value for the key at each epoch until
    the maximum epoch trained in the cross-folds
    """
    num_folds = len(history[key])
    vals = []
    epoch = 0
    while True:
        avg_val = 0.
        num_vals = 0
        for i in range(num_folds):
            if len(history[key][i]) > epoch:
                num_vals += 1
                avg_val += history[key][i][epoch]
            # use the last value if we've run out of epochs for this fold
            else:
                avg_val += history[key][i][-1]
        if num_vals == 0:
            break
        else:
            vals.append(avg_val / num_folds)
            epoch += 1
    return np.array(vals)

def plot_cv(history, suffix, ax, cols = ['tab:blue', 'tab:orange'], **kwargs):
    train_mean = get_average(history, f"train_{suffix}")
    val_mean = get_average(history, f"val_{suffix}")

    num_folds = len(history[f"train_{suffix}"])

    for i in range(num_folds):
        ax.plot(
            history[f"train_{suffix}"][i],
            alpha = 0.5,
            label = "Train" if i == 0 else None,
            color = cols[0],
            linestyle = 'dashed',
            **kwargs
        )
        ax.plot(
            history[f"val_{suffix}"][i],
            alpha = 0.5,
            label = "Val" if i == 0 else None,
            color = cols[1],
            linestyle = 'dashed',
            **kwargs
        )
        # add red dot if was terminates
        if len(history[f"train_{suffix}"][i]) < len(train_mean):
            ax.plot(len(history[f"train_{suffix}"][i]) - 1, history[f"train_{suffix}"][i][-1], color='red', marker='o', alpha=0.3)
            ax.plot(len(history[f"val_{suffix}"][i]) - 1, history[f"val_{suffix}"][i][-1], color='red', marker='o', alpha=0.3)

    ax.plot(train_mean, label="Train (CV mean)", color=cols[0], **kwargs)
    ax.plot(val_mean, label="Val (CV mean)", color=cols[1], **kwargs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(suffix)
    ax.legend()


def load_hist(experiment_name):
    global _cfg
    return np.load(os.path.join(_cfg['experiment_directory'], f"{experiment_name}.npy"), allow_pickle=True).item()


def get_config():
    global _cfg
    return _cfg
