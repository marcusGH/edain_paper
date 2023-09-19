from src.lib import plotting
import matplotlib.pyplot as plt

def compare_hists(histories, labels, key, ax, cols=None):
    if cols is None:
        cols = ['black', 'tab:blue', 'tab:orange', 'tab:green',
                'tab:red', 'tab:purple', 'black', 'grey']
        linestyles = ['solid', 'dashed', 'dotted', 'dashdot',
                      (5, (10, 3)), (0, (5, 10))]
    assert len(histories) <= len(
        cols) and "Not enough colours specified"

    for hist, c, l, ls in zip(histories, cols, labels, linestyles):
        cv_avgs = plotting.get_average(hist, key)
        ax.plot(cv_avgs, label=f"k={l} ({cv_avgs[-1]:.4f})", color=c,
                alpha=1, linestyle=ls)
        ax.plot(len(cv_avgs) - 1, cv_avgs[-1], marker='o', color=c)
    ax.legend(title=f"Number of clusters (validation loss)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average cross-validation loss")

num_clusters = [2, 4, 6, 8, 10, 20, 30]

histories = []
for i in range(len(num_clusters)):
    histories.append(plotting.load_hist(f"mixture-clustering-tuning-{i}"))

plotting.update_plot_params()
fig, ax = plt.subplots(figsize=plotting.get_figsize(fraction=1.0))
compare_hists(histories, num_clusters, "val_loss", ax)
plotting.save_plot(fig, "mixture-clustering-tuning")
