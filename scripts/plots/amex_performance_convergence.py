from src.lib.plotting import (
    get_figsize,
    load_hist,
    get_confidence_interval,
    save_plot,
    get_average,
    update_plot_params,
)
import matplotlib.pyplot as plt
import pandas as pd

# %% load all the history objects for the methods being compared
hist_ss = load_hist("standard-scaling-no-time-1")
# baselines
hist_010 = load_hist("baseline-010")
hist_110 = load_hist("baseline-110")
hist_011 = load_hist("baseline-011")
hist_111 = load_hist("baseline-111")
hist_mixed = load_hist("mixture-clustering-tuning-5")
# Ablation studies
hist_edain_0110 = load_hist("EDAIN-ABLATION-0110")
hist_edain_0111 = load_hist("EDAIN-ABLATION-0111")
hist_edain_1110 = load_hist("EDAIN-ABLATION-1110")
hist_edain = load_hist("edain-preprocessing-1")

hist_edain_kl = load_hist("amex-edain-kl-preprocessing-1")
hist_dain = load_hist("amex-dain-preprocessing-1")
hist_bin = load_hist("amex-bin-preprocessing-1")
# additional histories for the paper
hist_no_preprocess = load_hist("no-preprocess-amex-RECENT")
hist_edain_local = load_hist("edain-local-aware-amex-RECENT")
hist_cdf_invert = load_hist("cdf-inversion-amex")
# Mc-Carty
hist_mccarty_01 = load_hist("mcCarter-amex-0.1")
hist_mccarty_1 = load_hist("mcCarter-amex-1")
hist_mccarty_10 = load_hist("mcCarter-amex-10")
hist_mccarty_100 = load_hist("mcCarter-amex-100")

# setup plotting parameters
linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (5, (10, 3)), (0, (5, 10)), (7, (7, 3)), (7, (7, 3, 1, 3))]
cols = ['black', 'tab:blue', 'black', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
names = ["No preprocessing", 'Z-score', 'Winsorize + Z-score', 'Z-score + YJ', 'Winsorize + Z-score + YJ', "CDF inversion", 'EDAIN-KL', 'EDAIN (global-aware)', 'EDAIN (local-aware)', 'DAIN', 'BIN', 'McCarter ($\\alpha=0.1$)', 'McCarter ($\\alpha=1$)', 'McCarter ($\\alpha=10$)', 'McCarter ($\\alpha=100$)']
histories = [hist_no_preprocess, hist_010, hist_110, hist_011, hist_111, hist_cdf_invert, hist_edain_kl, hist_edain, hist_edain_local, hist_dain, hist_bin, hist_mccarty_01, hist_mccarty_1, hist_mccarty_10, hist_mccarty_100]

# histories = [hist_edain_0110, hist_edain_0111, hist_edain_1110, hist_edain]
# names = ['global-aware EDAIN (shift+scale)', 'global-aware EDAIN (shift+scale+power transform)', 'global-aware EDAIN (outlier removal+shift+scale)', 'global-aware EDAIN (full)']

# for h in histories:
#     d = h['cli_arguments']
#     print("=====================================")
#     print(f"Preprocessing method: {h['experiment_name']}")
#     cmd_str = "python3 src/experiments/run_experiments.py"
#     for k in d.keys():
#         if d[k]:
#             cmd_str = f"{cmd_str} --{k}={d[k]}"
#     print(cmd_str)

# %% make validation loss plot
assert len(linestyles) == len(cols) == len(names) == len(histories)
fig, ax = plt.subplots(figsize=get_figsize(fraction=1.0, height_width_ratio=0.6))
for hist, c, ls, lab in zip(histories, cols, linestyles, names):
    cv_avgs = get_average(hist, key='val_loss')
    ax.plot(cv_avgs, label=f"{lab}", color=c, linestyle=ls)
    ax.plot(len(cv_avgs) - 1, cv_avgs[-1], marker='o', color=c)
ax.legend(title=f"Preprocessing method")
ax.set_xlabel("Epoch")
ax.set_ylabel("Average cross-validation loss")
fig.suptitle("Validation loss and convergence speed on AMEX dataset")
# save_plot(fig, "amex_performance_convergence")
# plt.close(fig)
plt.show()

# %% make amex metric plot
fig, ax = plt.subplots(figsize=get_figsize(fraction=1.0, height_width_ratio=0.6))
for hist, c, ls, lab in zip(histories, cols, linestyles, names):
    cv_avgs = get_average(hist, key='val_amex_metric')
    ax.plot(cv_avgs, label=f"{lab}", color=c, linestyle=ls)
    ax.plot(len(cv_avgs) - 1, cv_avgs[-1], marker='o', color=c)
ax.legend(title=f"Preprocessing method")
ax.set_xlabel("Epoch")
ax.set_ylabel("Average cross-validation AMEX metric")
fig.suptitle("AMEX metric and convergence speed on AMEX dataset")
# save_plot(fig, "amex_performance_convergence_metric")
# plt.close(fig)
plt.show()

# %% make a pandas dataframe for the table
df = pd.DataFrame(columns=['Method', 'Validation loss', 'AMEX metric'])
for i, (hist, lab) in enumerate(zip(histories, names)):
    m, s = get_confidence_interval(hist, key='val_loss')
    m2, s2 = get_confidence_interval(hist, key='val_amex_metric')
    df.loc[i] = [lab, f"${m:.4f} \pm {s:.4f}$", f"${m2:.4f} \pm {s2:.4f}$"]
print(df.to_latex(index=False))

# %% table of results per fold
update_plot_params()
fig, ax = plt.subplots(figsize=get_figsize(fraction=1.0, height_width_ratio=0.8))
for hist, c, ls, lab in zip(histories[1:], cols[1:], linestyles[1:], names[1:]):
    vals = get_confidence_interval(hist, key='val_loss', get_vals=True)
    ax.plot(range(1, 6), vals, label=f"{lab}", color=c, linestyle=ls, marker="x", markersize=9)
ax.legend(title=f"Normalization method", loc='upper left')
ax.set_xlabel("Fold")
ax.set_xticks(range(1, 6))
ax.set_ylabel("Cross-validation loss")
# fig.suptitle("Validation loss per fold on AMEX dataset")
save_plot(fig, "amex_performance_convergence_per_fold_paper_version")
plt.show()

#%% Off case for baseline 111
import numpy as np
loss_vals = get_confidence_interval(hist_111, key='val_loss', get_vals=True)
amex_vals = get_confidence_interval(hist_111, key='val_amex_metric', get_vals=True)
other_baselines = [hist_010, hist_110, hist_011]
# replace malignant fold
loss_vals[1] = np.mean([get_confidence_interval(h, 'val_loss', get_vals=True)[1] for h in other_baselines])
amex_vals[1] = np.mean([get_confidence_interval(h, 'val_amex_metric', get_vals=True)[1] for h in other_baselines])
print(f"${np.mean(loss_vals):.4f} \pm {1.96 * np.std(loss_vals) / np.sqrt(5):.4f}$", f"${np.mean(amex_vals):.4f} \pm {1.96 * np.std(amex_vals) / np.sqrt(5):.4f}$")
