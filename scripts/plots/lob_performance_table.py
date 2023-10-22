from src.lib.plotting import (
    get_figsize,
    load_hist,
    get_confidence_interval,
    save_plot,
    get_average,
update_plot_params
)
import matplotlib.pyplot as plt
import pandas as pd
import pprint

#%% load the histories
hist_bin = load_hist("LOB-BIN-experiment-final")
hist_dain = load_hist("LOB-DAIN-experiment-final")
hist_edain_local = load_hist("LOB-EDAIN-experiment-final-v1")
hist_edain_global = load_hist("LOB-EDAIN-global-experiment-final-v1")
hist_edain_kl = load_hist("LOB-EDAIN-KL-experiment-final-v1")
hist_min_max = load_hist("LOB-min-max-experiment-final")
hist_standard_scaling = load_hist("LOB-standard-scaling-experiment-final")
# additinal histories for the paper
hist_no_preprocess = load_hist("no-preprocess-lob-RECENT")
hist_cdf_invert = load_hist("cdf-inversion-lob-v2")

linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (5, (10, 3)), (0, (5, 10)), (0, (3, 1)), (0, (3, 10, 1, 10))]
cols = ['black', 'tab:blue', 'tab:brown', 'tab:green', 'tab:red', 'tab:purple', 'tab:orange', 'tab:pink']
histories = [hist_no_preprocess, hist_standard_scaling, hist_cdf_invert, hist_bin, hist_dain, hist_edain_local, hist_edain_global, hist_edain_kl]
names = ["No preprocessing", 'Standard scaling', 'CDF inversion', 'BIN', 'DAIN', 'EDAIN (local-aware)', 'EDAIN (global-aware)', 'EDAIN-KL']

for h in histories:
    d = h['cli_arguments']
    print("=====================================")
    print(f"Preprocessing method: {h['experiment_name']}")
    cmd_str = "python3 src/experiments/run_experiments.py"
    for k in d.keys():
        if d[k]:
            cmd_str = f"{cmd_str} --{k}={d[k]}"
    print(cmd_str)

# %% table of results
df = pd.DataFrame(columns=['Method', 'Cohen\'s Kappa', 'Average $F_1$-score'])
for i, (hist, lab) in enumerate(zip(histories, names)):
    m, s = get_confidence_interval(hist, key='kappa')
    m2, s2 = get_confidence_interval(hist, key='f1_avg')
    df.loc[i] = [lab, f"${m:.4f} \pm {s:.4f}$", f"${m2:.4f} \pm {s2:.4f}$"]
print(df.to_latex(index=False))

#%% table of results per fold
update_plot_params()
fig, axs = plt.subplots(nrows=2, figsize=get_figsize(fraction=1.0, height_width_ratio=1.6))
plt.subplots_adjust(hspace=0.3)
for hist, c, ls, lab in zip(histories, cols, linestyles, names):
    vals = get_confidence_interval(hist, key='kappa', get_vals=True)
    axs[0].plot(vals, label=f"{lab}", color=c, linestyle=ls, marker="x", markersize=9)
axs[0].legend(title=f"Preprocessing method", loc='center left', bbox_to_anchor=(1, 0.5))
axs[0].set_xlabel("Fold")
axs[0].set_xticks(range(9))
axs[0].set_title("Cohen's Kappa")
axs[0].set_ylabel("Cohen's Kappa")
for hist, c, ls, lab in zip(histories, cols, linestyles, names):
    vals = get_confidence_interval(hist, key='f1_avg', get_vals=True)
    axs[1].plot(vals, label=f"{lab}", color=c, linestyle=ls, marker="x", markersize=9)
axs[1].legend(title=f"Preprocessing method", loc='center left', bbox_to_anchor=(1, 0.5))
axs[1].set_xlabel("Fold")
axs[1].set_xticks(range(9))
axs[1].set_ylabel("Average macro-$F_1$-score")
axs[1].set_title("Average macro-$F_1$-score")
# save_plot(fig, "lob_performance_per_fold")
plt.show()