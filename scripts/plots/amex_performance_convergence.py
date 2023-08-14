from src.lib.plotting import (
    get_figsize,
    load_hist,
    get_confidence_interval,
    save_plot,
    get_average,
)
import matplotlib.pyplot as plt
import pandas as pd

# load all the history objects for the methods being compared
hist_ss = load_hist("standard-scaling-no-time-1")
hist_mixed = load_hist("mixture-clustering-tuning-5")
hist_edain = load_hist("edain-preprocessing-1")
hist_edain_kl = load_hist("amex-edain-kl-preprocessing-1")
hist_dain = load_hist("amex-dain-preprocessing-1")
hist_bin = load_hist("amex-bin-preprocessing-1")

# setup plotting parameters
linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (5, (10, 3)), (0, (5, 10))]
cols = ['black', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
names = ['Standard scaling', 'PREPMIX-CAPS (k=20)', 'EDAIN (global-aware)', 'EDAIN-KL', 'DAIN', 'BIN']
histories = [hist_ss, hist_mixed, hist_edain, hist_edain_kl, hist_dain, hist_bin]
assert len(linestyles) == len(cols) == len(names) == len(histories)

# make validation loss plot
fig, ax = plt.subplots(figsize=get_figsize(fraction=1.0, height_width_ratio=0.6))
for hist, c, ls, lab in zip(histories, cols, linestyles, names):
    cv_avgs = get_average(hist, key='val_loss')
    ax.plot(cv_avgs, label=f"{lab}", color=c, linestyle=ls)
    ax.plot(len(cv_avgs) - 1, cv_avgs[-1], marker='o', color=c)
ax.legend(title=f"Preprocessing method")
ax.set_xlabel("Epoch")
ax.set_ylabel("Average cross-validation loss")
fig.suptitle("Validation loss and convergence speed on AMEX dataset")
save_plot(fig, "amex_performance_convergence")
plt.close(fig)

# make amex metric plot
fig, ax = plt.subplots(figsize=get_figsize(fraction=1.0, height_width_ratio=0.6))
for hist, c, ls, lab in zip(histories, cols, linestyles, names):
    cv_avgs = get_average(hist, key='val_amex_metric')
    ax.plot(cv_avgs, label=f"{lab}", color=c, linestyle=ls)
    ax.plot(len(cv_avgs) - 1, cv_avgs[-1], marker='o', color=c)
ax.legend(title=f"Preprocessing method")
ax.set_xlabel("Epoch")
ax.set_ylabel("Average cross-validation AMEX metric")
fig.suptitle("AMEX metric and convergence speed on AMEX dataset")
save_plot(fig, "amex_performance_convergence_metric")
plt.close(fig)

# make a pandas dataframe for the table
df = pd.DataFrame(columns=['Method', 'Validation loss', 'AMEX metric'])
for i, (hist, lab) in enumerate(zip(histories, names)):
    m, s = get_confidence_interval(hist, key='val_loss')
    m2, s2 = get_confidence_interval(hist, key='val_amex_metric')
    df.loc[i] = [lab, f"${m:.4f} \pm {s:.4f}$", f"${m2:.4f} \pm {s2:.4f}$"]
print(df.to_latex(index=False))

# table of results per fold
fig, ax = plt.subplots(figsize=get_figsize(fraction=1.0, height_width_ratio=0.8))
for hist, c, ls, lab in zip(histories, cols, linestyles, names):
    vals = get_confidence_interval(hist, key='val_loss', get_vals=True)
    ax.plot(vals, label=f"{lab}", color=c, linestyle=ls, marker="x", markersize=9)
ax.legend(title=f"Preprocessing method")
ax.set_xlabel("Fold")
ax.set_xticks(range(5))
ax.set_ylabel("Cross-validation loss")
fig.suptitle("Validation loss per fold on AMEX dataset")
save_plot(fig, "amex_performance_convergence_per_fold")
