import matplotlib.pyplot as plt
import numpy as np
from src.lib.synthetic_data import SyntheticData
from src.lib.time_series_util import plot_autocorrelations
from scipy import stats

D = 2
T = 10
# lower bound, upper bound, and unormalized PDF
bounds = [(-3, 5), (-25, 20)]
f1 = lambda x: 10 * stats.norm.cdf(10 * x) * stats.norm.pdf(x)
# f2 = lambda x: (0.001 / np.abs(x + 3)) ** (1/100)
# f2 = lambda x: 10 * stats.norm.cdf(100 * x) * stats.norm.pdf(x)
f2 = lambda x: np.where(x > 3, np.exp(7-x), np.exp(x / 10) * (10 * np.sin(x) + 10))
f1 = lambda x: 10 * stats.norm.cdf(10 * x) * stats.norm.pdf(x) + 2 * np.where(x > 2.0, np.exp(x - 2), 0) * np.where(x < 2.1, np.exp(2.1 - x), 0)
f2 = lambda x: np.where(x > 3, np.exp(7-x), np.exp(x / 10) * (10 * np.sin(x) + 10))
# both of the two time-series will be q=3 and q=2, respecitvely
thetas = np.array([
    [-1., 0.5, -0.2, 0.8],
    [-1., 0.3, 0.9, 0.0]
    ])

synth_data = SyntheticData(
        dim_size = D,
        time_series_length = T,
        pdfs = [f1, f2],
        pdf_bounds = bounds,
        ar_q = 3,
        ar_thetas = thetas,
        cross_variables_cor_init_sigma=0.3,
        )

np.random.seed(42)
X, y = synth_data.generate_data(n = 10000, return_uniform=False)

fig, _ = plot_autocorrelations(X[:, 0, :], thetas[0, :], 1.0)
plt.show()
fig, _ = plot_autocorrelations(X[:, 1, :], thetas[1, :], 1.0)
plt.show()
