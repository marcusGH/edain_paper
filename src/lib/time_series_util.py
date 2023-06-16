import numpy as np
import matplotlib.pyplot as plt

def sample_acvs(xs, tau):
    x_bar = np.mean(xs)
    N = len(xs)
    autoCov = 0
    for i in np.arange(0, N-tau):
        autoCov += ((xs[i+tau])-x_bar)*(xs[i]-x_bar)
    return (1/(N-1))*autoCov

def ar_acvs(tau, thetas, sigma_noise):
    # assumed the first element is -1
    if thetas[0] != -1:
        raise ValueError("The first theta of the AR(q) params is not -1")

    q = thetas.shape[0] - 1
    s_tau = 0
    for j in range(q - tau + 1):
        s_tau += thetas[j] * thetas[j + tau]
    return s_tau * sigma_noise ** 2

def plot_autocorrelations(Xs, thetas, sigma_noise):
    """
    Compares theoretical autocorrelation sequence and sample accs

    :param Xs: np.ndarray of shape (N, T)
    :param thetas: np.ndarray of shape (q+1,)
    :param sigma_noise: float
    """
    fig, axs = plt.subplots(ncols = 3, figsize = (15, 5))

    # histogram of samples just for reference
    axs[0].hist(Xs.reshape((-1)), bins=150, density=True)
    axs[0].set_title("Histogram of all samples across time")

    # theoretical one on the left
    acvs = []
    for tau in range(Xs.shape[1] - 1):
        acvs.append(ar_acvs(tau, thetas, sigma_noise))
    # rho_tau = s_tau / s_0
    acvs = np.array(acvs) / acvs[0]
    axs[1].bar(np.arange(len(acvs)), acvs, width = 0.2)
    axs[1].axhline(0, color='black')
    axs[1].set_title("Theoretical MA(q) autocorrelation sequence")
    axs[1].set_xlabel("Tau")
    axs[1].set_ylabel("Autocorrelation")

    # sample accs on the right
    acvs = []
    for tau in range(Xs.shape[1] - 1):
        acvs.append(np.mean([sample_acvs(Xs[i, :], tau) for i in range(Xs.shape[0])]))
    # rho_tau = s_tau / s_0
    acvs = np.array(acvs) / acvs[0]
    axs[2].bar(np.arange(len(acvs)), acvs, width = 0.2)
    axs[2].axhline(0, color='black')
    axs[2].set_title("Sample autocorrelation sequence")
    axs[2].set_xlabel("Tau")
    axs[2].set_ylabel("Autocorrelation")
    return fig, axs

