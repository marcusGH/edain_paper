from src.lib.plotting import (
    get_figsize,
    load_hist,
    save_plot,
    get_average,
    update_plot_params,
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy import stats
import  os

hist_baseline_010 = load_hist("HPC-baseline-010")
hist_baseline_110 = load_hist("HPC-baseline-110")
hist_baseline_011 = load_hist("HPC-baseline-011")
hist_baseline_111 = load_hist("HPC-baseline-111")
hist_edain_global = load_hist("HPC-edain-global-2")
hist_edain_local = load_hist("HPC-edain-local-2")
hist_kdit_1 = load_hist("HPC-KDIT-0.1")
hist_kdit_2 = load_hist("HPC-KDIT-1")
hist_kdit_3 = load_hist("HPC-KDIT-10")
hist_kdit_4 = load_hist("HPC-KDIT-100")

histories = [hist_baseline_010, hist_baseline_110, hist_baseline_011, hist_baseline_111, hist_edain_global, hist_edain_local, hist_kdit_1, hist_kdit_2, hist_kdit_3, hist_kdit_4]
names = ["St. scaling", "Winsorize + St. scaling", "St. scaling + YJ", "Winsorize + St. scaling + YJ", "edain_global", "EDAIN (local-aware)", "KDIT (0.1)", "KDIT (1.0)", "KDIT (10.0)", "KDIT (100.0)"]


def get_confidence_interval(history, key, min_val='val_accuracy', get_vals=False):
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
            vals.append(history[key][i][np.argmax(history[min_val][i][:6])])
    # convert to numpy array and compute CI
    vals = np.array(vals)
    if get_vals:
        return vals
    else:
        return np.mean(vals), np.std(vals) * 1.96 / np.sqrt(len(vals))

df = pd.DataFrame(columns=['Method', 'Validation loss', 'Validation ACC'])
for i, (hist, lab) in enumerate(zip(histories, names)):
    m, s = get_confidence_interval(hist, key='val_loss')
    m2, s2 = get_confidence_interval(hist, key='val_accuracy')
    df.loc[i] = [lab, f"${m:.4f} \pm {s:.4f}$", f"${m2*100:.2f}\% \pm {s2*100:.2f}\%$"]
print(df.to_latex(index=False))

## Function for the one-sided sign test
def sign_test(arr1, arr2, mid_pval=True):
    ## Compare length of arrays
    if len(arr1) != len(arr2):
        raise ValueError(
            "Arrays must have the same length for the sign test.")
    ## Calculate length
    n = len(arr1)
    ## Calculate the signs
    signs = [1 if arr1[i] > arr2[i] else -1 for i in range(n)]
    num_positive = sum(1 for sign in signs if sign > 0)
    num_negative = sum(1 for sign in signs if sign < 0)
    # Calculate the test statistic
    p = (num_positive - num_negative) / (
                num_positive + num_negative)
    # Calculate p-value for arr1 having a higher mean using a binomial distribution
    if mid_pval:
        p_value = np.mean(
            binom.cdf([num_negative, num_negative - 1], n, 0.5))
    else:
        p_value = binom.cdf(num_negative, n, 0.5)
    return p, p_value


## Function for the one-sided t-test
def one_sided_t_test(arr1, arr2, equal_var=True):
    ## Perform a t-test
    t_statistic, two_sided_p_value = stats.ttest_ind(arr1, arr2,
                                                     equal_var=equal_var)
    ## Calculate the one-sided p-value for the mean of arr1 > mean of arr2
    one_sided_p_value = two_sided_p_value / 2 if t_statistic > 0 else 1 - two_sided_p_value / 2
    ## Return results
    return t_statistic, one_sided_p_value


edain_global_vals = get_confidence_interval(hist_edain_global, key='val_accuracy', get_vals=True)
baseline_011 = get_confidence_interval(hist_baseline_011, key='val_accuracy', get_vals=True)
print("==========================")
print("EDAIN vs. Baseline 0.11")
print(edain_global_vals, baseline_011)
print(np.mean(edain_global_vals), np.mean(baseline_011))
print(sign_test(edain_global_vals, baseline_011))
print("==========================")

#%%
## List of experiments for Amex
experiments_amex = [
    "standard-scaling-no-time-1",
    "mixture-clustering-tuning-5",
    "edain-preprocessing-1",
    "amex-edain-kl-preprocessing-1",
    "amex-dain-preprocessing-1",
    "amex-bin-preprocessing-1",
    "no-preprocess-amex-RECENT",
    "edain-local-aware-amex-RECENT",
    "cdf-inversion-amex"
]

## Amex results
amex_res = {}
for experiment_name in experiments_amex:
    amex_res[experiment_name] = np.load(
        os.path.join("experiment-results-v2",
                     f"{experiment_name}.npy"),
        allow_pickle=True).item()

## Obtain BCE loss and Amex metric
amex_loss = {};
amex_metric = {}
for experiment_name in experiments_amex:
    amex_loss[experiment_name] = get_confidence_interval(
        amex_res[experiment_name], key='val_loss', get_vals=True)
    amex_metric[experiment_name] = get_confidence_interval(
        amex_res[experiment_name], key='val_amex_metric',
        get_vals=True)

## Sign tests against "edain-preprocessing-1"
for experiment_name in experiments_amex:
    if experiment_name != "edain-preprocessing-1":
        print(
            f"Sign test for {experiment_name} against edain-preprocessing-1")
        print(
            f"BCE loss: {sign_test(amex_loss[experiment_name], amex_loss['edain-preprocessing-1'])[1]}")
        print(
            f"Amex metric: {sign_test(amex_metric['edain-preprocessing-1'], amex_metric[experiment_name])[1]}")
        print()

## Obtain CIs for BCE loss and Amex metric
amex_loss = {};
amex_metric = {}
for experiment_name in experiments_amex:
    amex_loss[experiment_name] = get_confidence_interval(
        amex_res[experiment_name], key='val_loss', get_vals=False)
    amex_metric[experiment_name] = get_confidence_interval(
        amex_res[experiment_name], key='val_amex_metric',
        get_vals=False)

print(amex_loss)
print(amex_metric)

## List of experiments for LOB
experiments_lob = [
    "LOB-BIN-experiment-final",
    "LOB-DAIN-experiment-final",
    "LOB-EDAIN-experiment-final-v1",
    "LOB-EDAIN-global-experiment-final-v1",
    "LOB-EDAIN-KL-experiment-final-v1",
    "LOB-min-max-experiment-final",
    "LOB-standard-scaling-experiment-final",
    "no-preprocess-lob-RECENT",
    "cdf-inversion-lob-v2"
]

## LOB
lob_res = {}
for experiment_name in experiments_lob:
    lob_res[experiment_name] = np.load(
        os.path.join("experiment-results-v2",
                     f"{experiment_name}.npy"),
        allow_pickle=True).item()

## Obtain F1 and Cohen's Kappa
lob_f1 = {};
lob_cohen = {}
for experiment_name in experiments_lob:
    lob_f1[experiment_name] = get_confidence_interval(
        lob_res[experiment_name], key='f1_avg', get_vals=True)
    lob_cohen[experiment_name] = get_confidence_interval(
        lob_res[experiment_name], key='kappa', get_vals=True)

## Sign tests against "LOB-EDAIN-experiment-final-v1"
for experiment_name in experiments_lob:
    if experiment_name != "LOB-EDAIN-experiment-final-v1":
        print(
            f"Sign test for {experiment_name} against LOB-EDAIN-experiment-final-v1")
        print(
            f"F1: {sign_test(lob_f1['LOB-EDAIN-experiment-final-v1'], lob_f1[experiment_name])[1]}")
        print(
            f"Cohen's Kappa: {sign_test(lob_cohen['LOB-EDAIN-experiment-final-v1'], lob_cohen[experiment_name])[1]}")
        print()

## Obtain F1 and Cohen's Kappa
lob_f1 = {};
lob_cohen = {}
for experiment_name in experiments_lob:
    lob_f1[experiment_name] = get_confidence_interval(
        lob_res[experiment_name], key='f1_avg', get_vals=False)
    lob_cohen[experiment_name] = get_confidence_interval(
        lob_res[experiment_name], key='kappa', get_vals=False)

print(lob_f1)
print(lob_cohen)

## List of experiments for synthetic data
experiments_synthetic = names

## Synthetic data
synt_res = {n : h for (n, h) in zip(names, histories)}

## Obtain BCE loss and binary accuracy
synt_loss = {}
synt_acc = {}
for experiment_name in experiments_synthetic:
    synt_loss[experiment_name] = synt_res[experiment_name][
        'val_loss']
    synt_acc[experiment_name] = synt_res[experiment_name][
        'val_accuracy']

## Sign tests against "edain_global"
for experiment_name in experiments_synthetic:
    if experiment_name != "edain_global":
        print(
            f"Sign test for {experiment_name} against edain_global")
        print(
            f"BCE loss: {sign_test(synt_loss[experiment_name], synt_loss['edain_global'])[1]}")
        print(
            f"Binary accuracy: {sign_test(synt_acc['edain_global'], synt_acc[experiment_name])[1]}")
        print()

## Obtain updated confidence intervals
for experiment_name in experiments_synthetic:
    synt_loss[experiment_name] = (
    np.mean(synt_res[experiment_name]['val_loss']),
    1.96 * np.std(synt_res[experiment_name]['val_loss']) / np.sqrt(
        len(synt_res[experiment_name]['val_loss'])))
    synt_acc[experiment_name] = (
    np.mean(synt_res[experiment_name]['val_accs']),
    1.96 * np.std(synt_res[experiment_name]['val_accs']) / np.sqrt(
        len(synt_res[experiment_name]['val_accs'])))

print(synt_loss)
print(synt_acc)