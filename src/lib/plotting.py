import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import pathlib

if os.path.isfile("config.yaml"):
    with open("config.yaml") as f:
        _cfg = yaml.load(f, Loader=yaml.FullLoader)
elif os.path.isfile(os.path.join("..", "config.yaml")):
    with open(os.path.join("..", "config.yaml")) as f:
        _cfg = yaml.load(f, Loader=yaml.FullLoader)
else:
    f = os.path.join(pathlib.Path(__file__).parent.parent.parent.resolve(), "config.yaml")
    if os.path.isfile(f):
        with open(f) as f:
            _cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise FileNotFoundError("Unable to locate configuration file: config.yaml")

def update_plot_params(**kwargs):
    params = {
        "text.usetex": True,
        "font.family": "serif",
        'font.serif': ['Computer Modern'],
        # 'font' : {'family': 'serif', 'serif': ['Computer Modern']},
        "axes.labelsize": 9,
        "font.size": 11,
        "legend.fontsize": 7,
        "legend.title_fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }
    # replace specified key-word arguments before updating
    for k, v in kwargs.items():
        params[k] = v
    plt.rcParams.update(params)

def get_figsize(width=418.25555, fraction=1.0, height_width_ratio=(5 ** .5 - 1)/2):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Doing \the\textwidth in Latex gives textwidth=418.25555pt
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * height_width_ratio

    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim

def save_plot(fig, plot_name):
    global _cfg
    save_path = os.path.join(__file__.split("src")[0], _cfg["plot_output_dir"], f"{plot_name}.pdf")
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.01)

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

def get_confidence_interval(history, key, min_val='val_loss', get_vals=False):
    """
    Returns a tuple of the mean, and plus/minus 95% confidence interval
    Picks the epoch with the lowest validation loss to determine
    what epoch to stop at
    """
    # the input is a history from the LOB dataset
    if 'split_results' in history.keys():
        vals = []
        for i in range(len(history['split_results'])):
            min_key_vals = [v[min_val] for v in history['split_results'][i]]
            min_val_idx = np.argmin(min_key_vals)
            # fetch the value
            vals.append(history['split_results'][i][min_val_idx][key])

    # the history comes from the Amex dataset
    else:
        num_folds = len(history[key])
        vals = []
        for i in range(num_folds):
            vals.append(history[key][i][np.argmin(history[min_val][i])])
    # convert to numpy array and compute CI
    vals = np.array(vals)
    if get_vals:
        return vals
    else:
        return np.mean(vals), np.std(vals) * 1.96 / np.sqrt(len(vals))


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
