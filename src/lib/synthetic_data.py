import scipy
from scipy import stats
import numpy as np
import pandas as pd

class SyntheticData:

    def __init__(self, dim_size, time_series_length, pdfs, pdf_bounds,
                 ar_q, ar_thetas, ar_sigma_noises=None,
                 cross_variables_cor_init_sigma=None,
                 pdf_approximation_number_of_samples=20_000,
                 response_noise_sigma=1e-3, response_thresh=0.5,
                 response_beta_sigma=1.0):
        """
        TODO

        :param response_thresh: The treshold for labelling response as 1.
                                The expected number of negative examples is
                                equal to this parameter.
        """

        # validate parameters
        if len(pdfs) != dim_size:
            raise ValueError("The number of provided PDFs does not match the number of dimensions")
        if len(pdf_bounds) != dim_size:
            raise ValueError("The number of provided PDF bounds does not match the number of dimensions")
        if ar_q >= time_series_length:
            raise ValueError("q must be less than T")
        if ar_thetas.shape != (dim_size, ar_q + 1):
            raise ValueError("The MA thetas must be of shape (D, q+1)")

        # general parameters
        self.D = dim_size
        self.T = time_series_length

        # parameters for converting the uniform samples to arbitrary distributions
        self.pdfs = pdfs
        self.pdf_bounds = pdf_bounds
        self.pdf_num_samples = pdf_approximation_number_of_samples
        self.pdf_cache = [None] * dim_size

        # parameters for synthesizing the covariance matrix Sigma
        self.q = ar_q
        self.thetas = ar_thetas
        self.ar_sigma_noises = ar_sigma_noises if ar_sigma_noises is not None else np.ones((self.D,))
        self.cor_init_sigma = cross_variables_cor_init_sigma
        self.sigma = None

        # parameters used for synthesizing the response
        self.respose_thresh = response_thresh
        self.response_noise_sigma = response_noise_sigma
        self.betas = np.random.normal(loc=0.0, scale=response_beta_sigma, size=(self.D * self.T,))
        # Ignoring the correlation between the Us, we have
        # E( sum(betas @ us) + epsilon ) = sum( E[beta_i] * E[U_i] ) = 1/2 * sum(E[beta_i])
        #                                = DT/2 E[beta_0] =: 0.5
        # We want the above expectation to be equal to 0.5 such that half the responses are positive
        # with the default threshold. Therefore, we recenter the beta expectation to 1/DT
        self.betas = self.betas - np.mean(self.betas) + 1 / (self.D * self.T)

    def _sample_pdf(self, f_idx, us):
        """
        Generates samples xs from the f_idx'th pdf such that F(xs) = us

        :param f_idx: integer indexing into self.pdfs
        :param us: np.ndarray of shape (num_samples,). The number of
                   samples returned will match this
        :param num_integration_samples: number of samples used to approximate F(x)
        :returns: np.ndarray of same shape as us
        """
        cache = self.pdf_cache[f_idx]
        # we also need to redo the cache if the number of samples changed
        if cache is None:
            A, B = self.pdf_bounds[f_idx]
            xs = np.linspace(A, B, self.pdf_num_samples)

            Fs = scipy.integrate.cumulative_trapezoid(self.pdfs[f_idx](xs), xs, initial = 0)
            # normalize the CDF so endpoint is at 1 (as pdf f might be unnormalized)
            Fs /= np.max(Fs)

            # shape (num_samples, 2):
            #   first column contains F(x)s
            #   second column contains xs
            cache = np.stack([Fs, xs]).T
            self.pdf_cache[f_idx] = cache

        # Find the idx such that F(x[idx]) = us
        cache_idx = np.searchsorted(cache[:, 0], us, side = 'right')
        samples = cache[cache_idx, 1]
        return samples

    def _create_responses(self, us):
        """
        :param us: np.ndarray of shape (n, D, T)
        """
        # number of responses to create
        n = us.shape[0]
        epsilons = np.random.normal(scale=self.response_noise_sigma, size=(n,))
        ys = (us.reshape((n, -1)) @ self.betas) + epsilons
        # print(f"epsilon {np.mean(epsilons)}")
        # print(f"betas {np.mean(self.betas)}")
        # print(f"us {np.mean(us)}")
        # print(f"us @ betas + epsilon {np.mean(ys)}")
        # These are normally distributed with mean 0
        # plt.hist(ys, bins = 200)
        # plt.show()
        return np.float32(ys  > self.respose_thresh)


    def _sample_correlated_uniforms(self, n):
        """
        Returns an array of uniform random variables correlated "like" sigma, where sigma
        is set up initializing self.mvn_sampler

        :param n: integer specifying the number of sets of (D, T) samples to generates
        :param sigma: np.ndarray of shape(D * T, D * T) specifying the covariance relation
        :return: np.ndarray of shape (n, D, T)
        """
        mvn_sampler = stats.multivariate_normal(mean = np.zeros(self.D * self.T), cov = self.sigma, allow_singular=True)
        # shape (n, D * T)
        norm_samples = mvn_sampler.rvs(n)
        u_samples = np.zeros((n, self.D * self.T))
        for i in range(self.D * self.T):
            # the marginal distribution of the MVN will have this std
            norm_obj = stats.norm(loc = 0., scale = np.sqrt(self.sigma[i, i]))
            # The CDF then gives uniform random samples
            u_samples[:, i] = norm_obj.cdf(norm_samples[:, i])

        # shape (n, D, T)
        u_samples = u_samples.reshape((n, self.D, self.T))
        return u_samples

    def _isPositiveDefinite(self, A):
        try:
            _ = np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    def _find_closest_psd_matrix(self, A):
        """Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        Author: Ahmed Fasih [3]

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

        [3] https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194
        """

        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if self._isPositiveDefinite(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not self._isPositiveDefinite(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3

    def _ar_acvs(self, tau, thetas, sigma_noise):
        # assumed the first element is -1
        if thetas[0] != -1:
            raise ValueError("The first theta of the AR(q) params is not -1")

        q = thetas.shape[0] - 1
        s_tau = 0
        for j in range(q - tau + 1):
            s_tau += thetas[j] * thetas[j + tau]
        return s_tau * sigma_noise ** 2

    def _create_covariance_matrix(self, thetas):
        if self.cor_init_sigma is not None:
            A = np.random.normal(loc = 0, scale = self.cor_init_sigma, size = (self.D * self.T, self.D * self.T))
            A = np.tril(A)
        else:
            A = np.zeros((self.D * self.T, self.D * self.T))

        # When we later sample MVN of shape (DT), we will unroll
        # one row at a time:
        # > np.arange(1, 11).reshape((D, T))
        # array([[ 1,  2,  3,  4,  5],
        #        [ 6,  7,  8,  9, 10]])
        # Therefore, to get (D, T) indexing, we use (d * T + i) when
        # indexing into A. We also want to make a lower-triangular
        # matrix, so the highest index should come first
        for d in range(self.D):
            # all possible lags
            for tau in range(self.T):
                # all valid starting entries
                for i in range(self.T - tau):
                    # we are setting the correlation between U[d,i] and U[d,i+tau]
                    A[(d * self.T + i) + tau , d * self.T + i] =\
                            self._ar_acvs(tau, self.thetas[d, :], self.ar_sigma_noises[d])
        # to preserve the diagonals, reset them after adding the transpose
        sigma = A + A.T
        np.fill_diagonal(sigma, np.diag(A))
        return self._find_closest_psd_matrix(sigma)

    def uniform_to_pdf_samples(self, us):
        n = us.shape[0]
        Xs = np.zeros((n, self.D, self.T))
        for f_idx in range(self.D):
            Xs[:, f_idx, :] = self._sample_pdf(f_idx, us[:, f_idx, :].reshape(-1)).reshape((n, self.T))
        return Xs

    def generate_data(self, n, return_uniform = False, random_state = None):
        """
        Generates correlated Xs and corresponding responses y

        :return: tuple (X, y) where X is an np.ndarray of shape (n, D, T) and y is of shape (n,)
        """

        if self.sigma is None:
            self.sigma = self._create_covariance_matrix(self.thetas)

        if random_state is not None:
            np.random.seed(random_state)

        # shape (n, D, T)
        us = self._sample_correlated_uniforms(n)
        # shape (n,)
        ys = self._create_responses(us)

        # corrupt the data using our PDFs and u samples
        Xs = np.zeros((n, self.D, self.T))
        for f_idx in range(self.D):
            Xs[:, f_idx, :] = self._sample_pdf(f_idx, us[:, f_idx, :].reshape(-1)).reshape((n, self.T))

        if return_uniform:
            return np.float32(us), ys
        else:
            return np.float32(Xs), ys
