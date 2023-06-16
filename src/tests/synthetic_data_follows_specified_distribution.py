import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import quad
from scipy import stats
from src.lib.synthetic_data import SyntheticData

def plot_hist_with_pdf(xs, f):
    fig, ax = plt.subplots(figsize=(5,5))

    A, B = np.min(xs), np.max(xs)

    C = quad(f, A, B)[0]
    x = np.linspace(A, B, 10000)

    ax.hist(xs, bins = 200, density = True)
    ax.plot(x, f(x) / C, color='red')
    return fig, ax

D = 2
T = 6
# lower bound, upper bound, and unormalized PDF
bounds = [(-3, 5), (-25, 20)]
f1 = lambda x: 10 * stats.norm.cdf(10 * x) * stats.norm.pdf(x)
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
        cross_variables_cor_init_sigma=None,
        )

np.random.seed(42)
X, y = synth_data.generate_data(n = 10000, return_uniform=False)

fig, _ = plot_hist_with_pdf(X[:, 0, 0], f1)
plt.show()
fig, _ = plot_hist_with_pdf(X[:, 1, 0], f2)
plt.show()
