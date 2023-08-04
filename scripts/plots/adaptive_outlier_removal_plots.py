import matplotlib.pyplot as plt
import numpy as np
from src.lib.plotting import (
    update_plot_params,
    get_figsize,
    save_plot,
)

update_plot_params()

fig, ax = plt.subplots(ncols=2, figsize=get_figsize(fraction=1.0, height_width_ratio=0.5))

def f(xs, alpha, beta):
    assert (alpha >= 0) and (alpha <= 1)
    assert beta > 0

    xs = np.array(xs)
    return (1 - alpha) * xs + alpha * beta * np.tanh(xs / beta)

xs = np.linspace(-12, 12, 5000)

for i, beta in enumerate([1.0, 5.0]):
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (5, (10, 3))]
    for alpha, ls in zip(reversed([0.0, 0.25, 0.5, 0.75, 1.0]), linestyles):
        ax[i].plot(xs, f(xs, alpha, beta), label=f"$\\alpha'={alpha}$", linestyle=ls)
    ax[i].legend(title="Winsorization ratio")
    ax[i].set_xlabel("$x$")
    ax[i].set_ylabel("$h_1(x)$")
    ax[i].set_title(f"Threshold $\\beta'={beta}$")
plt.tight_layout(pad=0.0)
save_plot(fig, "adaptive_outlier_removal")

