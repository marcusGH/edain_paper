import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import sklearn
import scipy
from scipy.integrate import quad, cumulative_trapezoid
from scipy import stats
from scipy.stats import norm
from src.lib.synthetic_data import SyntheticData
from src.experiments.static_preprocessing_methods.standard_scaling import StandardScalerTimeSeries
from src.lib import experimentation
from src.models import basic_grunet
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
from tqdm.auto import tqdm

with open(os.path.join("config.yaml")) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

def evaluate_model(datasets, preprocess_init_fn=None):
    tlosses = []
    vlosses = []
    tmetrics = []
    vmetrics = []
    accs = []
    num_epochs = []

    for X, y in tqdm(datasets):
        N = X.shape[0]

        model = basic_grunet.GRUNetBasic(X.shape[2], 32, 2, 2, 0)
        loss_fn = F.binary_cross_entropy
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 7], gamma=0.1)
        early_stopper = experimentation.EarlyStopper(patience=5)

        # train-val split
        X_train, y_train = X[:int(N * 0.8)], y[:int(N * 0.8)]
        X_val, y_val = X[int(N * 0.8):], y[int(N * 0.8):]

        # preprocess the dataset if provided with a method
        if preprocess_init_fn is not None:
            preprocess = preprocess_init_fn()
            X_train = preprocess.fit_transform(X_train, y_train)
            X_val = preprocess.transform(X_val)

        train_loader = torch.utils.data.DataLoader(
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(X_train).type(torch.float32),
                    torch.from_numpy(y_train).type(torch.float32)),
                batch_size=128, shuffle = True)
        val_loader = torch.utils.data.DataLoader(
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(X_val).type(torch.float32),
                    torch.from_numpy(y_val).type(torch.float32)),
                batch_size=128, shuffle = True)

        history = experimentation.fit_model(model, loss_fn, train_loader, val_loader, optimizer, num_epochs=30, early_stopper=early_stopper, scheduler=scheduler, verbose=False)

        # compute the classification binary accuracy
        val_preds = []
        val_labs = []
        for x_, y_ in val_loader:
            val_preds.append(model(x_.cuda()).detach().cpu().numpy())
            val_labs.append(y_.cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_labs = np.concatenate(val_labs)

        vlosses.append(history['val_loss'][-1])
        tlosses.append(history['train_loss'][-1])
        vmetrics.append(history['val_amex_metric'][-1])
        tmetrics.append(history['train_amex_metric'][-1])
        accs.append(np.mean(np.where(val_preds > 0.5, 1.0, 0.0) == val_labs))
        num_epochs.append(len(history['train_loss']))

    return {
        'val_loss' : np.array(vlosses),
        'val_amex_metric' : np.array(vmetrics),
        'train_loss' : np.array(tlosses),
        'train_amex_metric' : np.array(tmetrics),
        'val_accs' : np.array(accs),
        'num_epochs' : np.array(num_epochs),
    }

#############################
### Experiment parameters ###
#############################

D = 2
T = 6
# lower bound, upper bound, and unormalized PDF
bounds = [(-3, 8), (-25, 20)]
f1 = lambda x: 10 * stats.norm.cdf(10 * x) * stats.norm.pdf(x) + 3 * np.where(x > 6, np.exp(x - 6), 0) * np.where(x < 6.2, np.exp(6.2 - x), 0)
f2 = lambda x: np.where(x > 3, np.exp(7-x), np.exp(x / 10) * (10 * np.sin(x) + 10))
# both of the two time-series will be q=3 and q=2, respecitvely
thetas = np.array([
    [-1., 0.5, -0.2, 0.8],
    [-1., 0.3, 0.9, 0.0]
])
CROSS_VAR_SIGMA = 1.4
RESPONSE_NOISE_SIGMA = 0.5
RESPONSE_BETA_SIGMA = 2.0
RANDOM_STATE = 42
NUM_DATASETS = 50

########################
### Experiment setup ###
########################

synth_data_irregular = SyntheticData(
    dim_size = D,
    time_series_length = T,
    pdfs = [f1, f2],
    pdf_bounds = bounds,
    ar_q = thetas.shape[1] - 1,
    ar_thetas = thetas,
    cross_variables_cor_init_sigma=CROSS_VAR_SIGMA,
    response_noise_sigma=RESPONSE_NOISE_SIGMA,
    response_beta_sigma=RESPONSE_BETA_SIGMA,
    random_state=RANDOM_STATE,
)

synth_data_norm = SyntheticData(
    dim_size = D,
    time_series_length=T,
    pdfs = [scipy.stats.norm.pdf, scipy.stats.norm.pdf],
    ar_q = thetas.shape[1] - 1,
    ar_thetas=thetas,
    pdf_bounds= [(-5, 5), (-5, 5)],
    cross_variables_cor_init_sigma=CROSS_VAR_SIGMA,
    response_noise_sigma=RESPONSE_NOISE_SIGMA,
    response_beta_sigma=RESPONSE_BETA_SIGMA,
    random_state=RANDOM_STATE,
)

def cdf_f1(x, A=-50, B=50, n=40000):
    # cache the integral computation
    if cdf_f1.Fs is None:
        xs = np.linspace(A, B, n)
        Fs = cumulative_trapezoid(f1(xs), xs, initial=0)
        Fs /= np.max(Fs)
        cdf_f1.Fs = Fs
    assert np.all(x > A) and np.all(x < B)
    # find the appropriate index by bucketing
    idx = np.array((x - A) * n / (B - A), dtype=np.int32)
    return cdf_f1.Fs[idx]
cdf_f1.Fs = None

def cdf_f2(x, A=-50, B=50, n=40000):
    # cache the integral computation
    if cdf_f2.Fs is None:
        xs = np.linspace(A, B, n)
        Fs = cumulative_trapezoid(f2(xs), xs, initial=0)
        Fs /= np.max(Fs)
        cdf_f2.Fs = Fs
    assert np.all(x > A) and np.all(x < B)
    # find the appropriate index by bucketing
    idx = np.array((x - A) * n / (B - A), dtype=np.int32)
    return cdf_f2.Fs[idx]
cdf_f2.Fs = None

if __name__ == "__main__":
    # setup the datasets
    norm_datasets = []
    irregular_datasets = []
    for i in tqdm(range(NUM_DATASETS), desc="Generating datasets"):
        irregular_datasets.append(synth_data_irregular.generate_data(n=10000, return_uniform=False, random_state=i))
        norm_datasets.append(synth_data_norm.generate_data(n=10000, return_uniform=False, random_state=i))

    # parse args to decide experiments
    exps = [int(x) for x in sys.argv[1:]]

    ###################################################
    ### Experiment 1: Performance on irregular data ###
    ###################################################

    if 1 in exps:
        print("Starting experiment 1: Performance on irregular data")
        results = evaluate_model(irregular_datasets)
        np.save(os.path.join(cfg['experiment_directory'], 'synth-data-irregular-data2.npy'), results)

    ###################################################
    ### Experiment 2: Performance on normal data    ###
    ###################################################

    if 2 in exps:
        print("Starting experiment 2: Performance on normal data")
        results = evaluate_model(norm_datasets)
        np.save(os.path.join(cfg['experiment_directory'], 'synth-data-normal-data2.npy'), results)

    ########################################################
    ### Experiment 3: Standard scaling on irregular data ###
    ########################################################

    if 3 in exps:
        print("Starting experiment 3: Standard scaling on irregular data")
        results = evaluate_model(irregular_datasets, preprocess_init_fn=lambda : StandardScalerTimeSeries(time_series_length=6))
        np.save(os.path.join(cfg['experiment_directory'], 'synth-data-standard-scaling2.npy'), results)

    ##################################################################
    ### Experiment 4: Undo corruption on irregular data using CDFs ###
    ##################################################################

    class UndoCorruptionTimeSeries(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
        def __init__(self, time_series_length = 6, epsilon=1E-4):
            self.epsilon = epsilon
            self.T = time_series_length

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            for t in range(self.T):
                # inverse normal CDF * original CDF
                X[:, t, 0] = norm.ppf(cdf_f1(X[:, t, 0]) * (1 - self.epsilon) + self.epsilon / 2)
                # inverse normal CDF * original CDF
                X[:, t, 1] = norm.ppf(cdf_f2(X[:, t, 1]) * (1 - self.epsilon) + self.epsilon / 2)
            return X

    if 4 in exps:
        print("Starting experiment 4: Undo transformation on irregular data")
        results = evaluate_model(irregular_datasets, preprocess_init_fn=lambda : UndoCorruptionTimeSeries())
        np.save(os.path.join(cfg['experiment_directory'], 'synth-data-undo-transform2.npy'), results)
