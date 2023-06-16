import matplotlib.pyplot as plt
import numpy as np
import scipy
from src.lib.synthetic_data import SyntheticData

D = 2
T = 6
# lower bound, upper bound, and unormalized PDF
bounds = [(-3, 5), (-4, 16)]
f1 = lambda x: 10 * scipy.stats.norm.cdf(10 * x) * scipy.stats.norm.pdf(x)
f2 = lambda x: scipy.stats.norm.pdf(x - 6)
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
        cross_variables_cor_init_sigma=1.0,
        )

np.random.seed(42)
X, y = synth_data.generate_data(n = 10000, return_uniform=True)

plt.hist(X[:, 0, 0])
plt.show()
