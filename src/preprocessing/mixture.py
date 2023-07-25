from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.preprocessing.static_transformations import MinMaxTimeSeries, TanhStandardScalerTimeSeries, StandardScalerTimeSeries
from threading import Thread
from tqdm.auto import tqdm

import copy
import numpy as np
import os
import scipy
import sklearn
import torch
import torch.nn.functional as F

# Temporary import
from src.models.basic_grunet import GRUNetBasic
import torch
from src.lib.experimentation import EarlyStopper, undo_min_max_corrupt_func, load_amex_numpy_data, fit_model
import yaml

_available_scalers = {
    'standard-scaler' : lambda : StandardScalerTimeSeries(),
    'min-max-scaler' : lambda : MinMaxTimeSeries(),
    'tanh-standard-scaler' : lambda : TanhStandardScalerTimeSeries(),
}

class MixedTransformsTimeSeries(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    def __init__(self, transforms_list, time_series_length = 13):
        """
        :param transforms_list: List of tuples on the form (var_list, sklearn.base.BaseEstimator)
        where var_list is a list of integers indicating which d in {0, 1, ..., D-1} along the third
        dimension of X that provided preprocessing transformer should be applied to
        Note that the transformations supplied should be able to fit (N, T, D)-dimensional data
        """
        self.vars = [x for (x, _) in transforms_list]
        self.transforms = [y() for (_, y) in transforms_list]
        self.all_vars = [j for sub in self.vars for j in sub]
        self.T = time_series_length

    def fit(self, X, y = None):
        """
        :param X: np.ndarray of shape (N, T, D)
        """
        # assert that initialised correctly
        D = X.shape[2]
        assert len(self.all_vars), len(list(set(self.all_vars))) == (D, D) and "No dupes"
        assert list(sorted(self.all_vars)) == list(range(D)) and "All elements included"

        for i in range(len(self.vars)):
            # only use the subset of variables specified, and use in shape (N, T * D')
            X_sub = X[:, :, self.vars[i]] #.reshape((X.shape[0], -1))
            self.transforms[i].fit(X_sub, y)
        return self

    def transform(self, X):
        for i in range(len(self.vars)):
            # only use the subset of variables specified, and use in shape (N, D')
            X_sub = X[:, :, self.vars[i]] #.reshape((X.shape[0], -1))
            X[:, :, self.vars[i]] = self.transforms[i].transform(X_sub)
        return X


def get_histogram(X, num_bins=1000):
    # Maps (N, T, D) array to (D, num_bins)
    D = X.shape[2]
    a = X.reshape((-1, D))
    hist = np.zeros((D, num_bins))
    # so that we get num_bins values when applying histogram
    bins = np.linspace(0, 1, num_bins + 1)

    for d in tqdm(range(D), desc="Binning data along each dimension"):
        hist[d, :] = np.histogram(a[:, d], bins)[0]
    return hist


def get_distribution_statistics(X, y=None, T=13, num_bins=1000):
    """
    Takes input array of shape (N, T, D) and returns an array of shape (D, L),
    where L is the number of statistics computed by this function. Currently, L = 6
    """
    scaler = MinMaxTimeSeries(time_series_length=T)
    D = X.shape[2]
    X_scaled = scaler.fit_transform(X, y)

    # bin all the data, required for certain statistics
    hist = get_histogram(X_scaled, num_bins) # (D, num_bins)
    # merge num_samples and time-axis, required for certain statistics
    X_flat = X_scaled.reshape((-1, D))       # (N', D)
    # collect all the statistics of shape (D,) here
    X_statistics = []

    # TODO: add manually updating progress bar for the statistics...
    # Statistic 1: skewness
    X_statistics.append(
        scipy.stats.skew(X_flat, axis=0, nan_policy='raise')
    )
    # Statistic 2: kurtosis
    X_statistics.append(
        scipy.stats.kurtosis(X_flat, axis=0, nan_policy='raise')
    )
    # Statistic 3: std
    X_statistics.append(
        np.std(X_flat, axis=0)
    )
    # Statistic 4: normalized location of highest mode
    X_statistics.append(
        np.argmax(hist, axis=1) / num_bins
    )
    # Statistic 5: normalized number of unique values
    X_statistics.append(
        np.apply_along_axis(lambda x : np.sum(x > 0), axis=1, arr=hist) / num_bins
    )
    # Statistic 6: number of values in highest frequent bin
    X_statistics.append(
        np.max(hist, axis=1)
    )

    return np.stack(X_statistics, axis=1)


def cluster_variables_with_statistics(X, k, y=None, stat_kwargs=None, **kmeans_kwargs):
    """
    Given an array of shape (N, T, D), returns a list of lists
    of k lists, constituting a partition of the {1, ..., D} variables
    """
    if stat_kwargs is None:
        stat_kwargs = {}
    D = X.shape[2]

    # get statistics for each variable, shape (D, L), where L number of statistics
    dist_stats = get_distribution_statistics(X, y, **stat_kwargs)
    # scale it before clustering
    feature_scaler = StandardScaler()
    dist_stats = feature_scaler.fit_transform(dist_stats)

    # Apply K-means clustering
    assert 'random_state' in kmeans_kwargs.keys()
    km = KMeans(n_clusters=k, **kmeans_kwargs)
    km.fit(dist_stats)

    # get the groupings
    groups = []
    for i in range(k):
        groups.append(list(np.arange(D)[km.labels_ == i]))
    return groups


def cluster_variables_with_kl_difference(X, y=None, **kwargs):
    """
    Given an array of shape (N, T, D), returns a list of lists
    of k lists, constituting a partition of the {1, ..., D} variables
    """
    raise NotImplementedError("Not implemented yet...")


def run_mixture_job(
        transform_list,
        dev,
        save_file_name,
        model_init_fn,
        optimizer_init_fn,
        scheduler_init_fn,
        early_stopper_init_fn,
        X,
        y,
        exp_cfg,
        random_state=42,
    ):
    """
    Trains a single model on 80% of the provided data, and validates it on the reaming 20%.
    A mixture of transformations, as specified by parameter transform_list is used for the
    experiment.

    :param save_path: location to save the history after finishing the job
    """
    num_cat = exp_cfg['num_categorical_features']

    # setup the dataset splits and fit and apply the scaler
    scaler = MixedTransformsTimeSeries(transform_list)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
    )
    X_train[:, :, num_cat:] = scaler.fit_transform(X_train[:, :, num_cat:], y_train)
    X_val[:, :, num_cat:] = scaler.transform(X_val[:, :, num_cat:])

    # create the data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
        ),
        **exp_cfg['data_loader']
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float(),
            ),
        **exp_cfg['data_loader']
    )

    # Setup loss function
    if exp_cfg['fit']['loss'] != 'bce':
        raise NotImplementedError("Loss not supported: " + exp_cfg['fit']['loss'])
    else:
        loss_fn = F.binary_cross_entropy

    # setup remaining objects
    model = model_init_fn()
    optimizer = optimizer_init_fn(model)
    scheduler = scheduler_init_fn(optimizer)
    early_stopper = early_stopper_init_fn()

    hist = fit_model(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopper=early_stopper,
        num_epochs=exp_cfg['mixture']['num_epochs_brute_force'],
        verbose=False,
        device_ids=dev,
    )

    np.save(os.path.join(exp_cfg['mixture']['cache_directory'], f"{save_file_name}.npy"), hist)
    print(f"Saved results for: {save_file_name}")


def create_mixture_job_args(variable_cluster_groups, exp_cfg):
    """
    Returns a list of tuples on the form:
    (name_of_transform, transform_list) where name_of_transform is either "baseline"
    or on the form "mixture-gid_<group ID>-scaler_<name of scaler>"
    """
    global _available_scalers
    config_scalers = exp_cfg['mixture']['transforms']
    num_groups = len(variable_cluster_groups)

    # as baseline, apply standard-scaling to all the variables
    base_transform_list = [
        (g, lambda : StandardScalerTimeSeries()) for g in variable_cluster_groups
    ]

    # setup all the job names and associated transform lists
    jobs = [('baseline', base_transform_list)]
    for i, group in enumerate(variable_cluster_groups):
        for scaler_name in _available_scalers.keys():
            # not in the config, so skip this even if available
            if scaler_name not in config_scalers or scaler_name == 'standard-scaler':
                continue

            transform_list = copy.deepcopy(base_transform_list)
            # change the ith transform with one of the available ones
            transform_list[i] = (
                transform_list[i][0],
                copy.deepcopy(_available_scalers[scaler_name])
            )
            job_name = f"mixture-gid_{i}-scaler_{scaler_name}"
            jobs.append((job_name, transform_list))

    # sanity check
    assert len(jobs) == 1 + (len(config_scalers) - int('standard-scaler' in config_scalers)) * num_groups

    return jobs


def run_parallel_mixture_jobs(
        job_list,
        devices_ids,
        **job_kwargs
    ):
    """
    The arguments to run_job are:
      * transform_list,   --\
      * dev,                | -- specified by positional args
      * save_file_name,   --/
      * model_init_fn,    -----\
      * optimizer_init_fn,     |
      * scheduler_init_fn,     |
      * early_stopper_init_fn, | -- part of job_kwargs
      * X,                     |
      * y,                     |
      * exp_cfg,               |
      * random_state=42,  ----/
    """
    exp_cfg = job_kwargs['exp_cfg']

    # keep track of all the threads spawned
    threads = []
    for i, (job_name, transform_list) in enumerate(job_list):
        dev_id = devices_ids[((i // exp_cfg['jobs_per_gpu']) % len(devices_ids))]
        threads.append(Thread(
            target=run_mixture_job,
            args=(transform_list, torch.device('cuda', dev_id), job_name),
            kwargs=job_kwargs,
        ))

    i = 0
    while i < len(threads):
        # start of all the threads
        for j in range(exp_cfg['jobs_per_gpu'] * len(devices_ids)):
            threads[i + j].run()
        # wait for them to finish
        for j in range(exp_cfg['jobs_per_gpu'] * len(devices_ids)):
            threads[i + j].join()
            print(f"Finished running job {i+j} / {len(threads)}")
        # then start of more jobs if any left
        i += exp_cfg['jobs_per_gpu'] * len(devices_ids)

def brute_force_preprocessing_mixture(**kwargs):
    # TODO: implement this
    #       this is [1/2] of the main driver and should be called to train and cache the brute force model
    global _available_scalers

    # setup the rest of the keyword arguments for the job runner method
    mixture_job_runner_kwargs = {
        'model_init_fn' : None,
        'optimizer_init_fn' : None,
        'scheduler_init_fn' : None,
        'early_stopper_init_fn' : None,
        'X' : None,
        'y' : None,
        'exp_cfg' : None,
        'save_path' : None,
        'random_state' : None,
    }
    raise NotImplementedError("Not implemented yet")


def get_optimal_mixture_transform_list(experiment_name):
    # TODO: implement this
    #       this is [2/2] of the main driver, and looks up the caches result for everything in cache
    #       folder matching experiment name prefix, then looks at the histories and constructs the
    #       optimal transform list, which should then be returned.
    #       (Note: this is the "conclude" block in the diagram)
    global _available_scalers
    raise NotImplementedError("Not implemented yet")


if __name__ == "__main__":
    print("Testing testing...")

    with open("src/experiments/configs/experiment-config-alpha.yaml", "r") as f:
        exp_cfg = yaml.load(f, Loader=yaml.FullLoader)

    model_init_fn = lambda : GRUNetBasic(
        num_cat_columns=exp_cfg['num_categorical_features'],
        **exp_cfg['gru_model']
    )
    optimizer_init_fn = lambda mod : torch.optim.Adam(
        mod.parameters(),
        lr=1e-3
    )
    scheduler_init_fn = lambda optim : torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optim,
        milestones=[3, 7],
        gamma=0.1,
    )
    early_stopper_init_fn = lambda : EarlyStopper()
    # load the data
    X, y = load_amex_numpy_data(
        split_data_dir=os.path.join("/home/silo1/mas322/amex-default-prediction/", 'derived', 'processed-splits'),
        fill_dict=exp_cfg['fill'],
        corrupt_func=lambda X, y: undo_min_max_corrupt_func(X, y, 42),
        num_categorical_features=exp_cfg['num_categorical_features']
    )

    transform_list = [
        (list(range(100)), lambda : StandardScalerTimeSeries()),
        (list(range(100, 177)), lambda : TanhStandardScalerTimeSeries()),
    ]

    # Test run mixture job
    run_mixture_job(
        transform_list,
        torch.device('cuda', 3),
        'test-filename',
        model_init_fn,
        optimizer_init_fn,
        scheduler_init_fn,
        early_stopper_init_fn,
        X,
        y,
        exp_cfg,
        random_state=42,
    )

