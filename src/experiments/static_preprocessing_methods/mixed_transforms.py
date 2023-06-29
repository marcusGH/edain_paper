import yaml
import sklearn
import os
import copy
import argparse
import numpy as np
import torch
from src.experiments.static_preprocessing_methods import winsorization
from src.experiments.static_preprocessing_methods.standard_scaling import StandardScalerTimeSeries
from src.experiments.static_preprocessing_methods.log_standard_scaling import LogStandardScalerTimeSeries
from src.experiments.static_preprocessing_methods.min_max_scaling import MinMaxTimeSeries
from src.experiments.static_preprocessing_methods.log_min_max_scaling import LogMinMaxTimeSeries
from src.experiments.static_preprocessing_methods.tanh_standard_scaling import TanhStandardScalerTimeSeries
from src.lib import experimentation
from sklearn import preprocessing

with open("config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

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

if __name__ == "__main__":
    import src.experiments.static_preprocessing_methods.experiment_setup as setup

    parser = argparse.ArgumentParser()

    # Use like:
    # python script.py -g 1 2 3 4
    parser.add_argument('-g', '--groups', nargs='+', help='<Required> Specify group', required=False, default=[])
    parser.add_argument('-b', '--baseline', action='store_true', required=False)
    parser.add_argument('-d', '--device', metavar='D', type=int, required=True)
    parser.add_argument('--optimal-loss-mix', action='store_true', required=False)

    # 1: skewed normal, possibly with outliers
    group1 = [26, 39, 57, 68, 123, 125, 130, 137, 145, 153]
    # 2: left and right tops
    group2 = [17, 19, 24, 46, 47, 52, 53]
    # 3: left-half declining continous Poisson
    group3 = [2, 7, 8, 9, 11, 29, 36, 40, 43, 44, 48, 50, 64, 67, 70, 71, 72, 76, 91, 127, 135, 136, 159]
    # 4: left-half like above, but with more mass in middle
    # 4.5: right-half of skewed normal, like above, but right-side
    group4 = [1, 20,  22, 30, 31, 38, 75, 104, 144, 147, 148, 149, 162] + [0, 150]
    # 5: one extreme value (high density spike)
    group5 = [6, 13, 14, 18, 33, 34, 41, 51, 58, 60, 61, 69, 73, 74, 77, 80, 81, 82, 87, 89, 90, 92, 94, 95, 96, 98, 99, 102, 107, 108, 109, 110, 111, 113, 114, 118, 119, 120, 124, 126, 134, 138, 141, 158, 161, 163, 166, 167, 168, 169, 172, 175]
    # 6: discrete-like normal with few spikes
    group6 = [3, 10, 23, 25, 37, 45, 49, 59, 63, 65, 66, 78, 79, 83, 93, 116, 133, 146, 151, 152, 176]
    # 7: multiple spikes/modes
    group7 = [4, 21, 32, 54, 55, 56, 62, 84, 85, 88, 97, 100, 112, 117, 122, 129, 132, 140, 143, 154, 155, 156, 157, 160, 170, 171, 174]
    # 8: Multiple modes/spikes, but looks skewed Gaussian overall
    # 8.5: multiple modes, but looks normal  overall
    group8 = [12, 131, 165, 173] + [16, 28]
    # 9: Other
    group9 = [5, 15, 27, 35, 42, 86, 101, 103, 105, 106, 115, 121, 128, 139, 142, 164]

    groups = [group1, group2, group3, group4, group5, group6, group7, group8, group9]

    args = parser.parse_args()
    do_baseline = args.baseline
    dev = args.device

    print(f"Performing experiment with arguments: {args}")

    # expected to be 1-indexed
    group_ids = [int(x) - 1 for x in args.groups]
    if len(group_ids) > 0:
        assert min(group_ids) >= 0 and "Group IDs invalid"
        assert max(group_ids) < len(groups) and "Group IDs invalid"

    baseline_transform_list = list(zip(groups, [lambda : StandardScalerTimeSeries()] * len(groups)))

    if do_baseline:
        preprocess_init = lambda : MixedTransformsTimeSeries(baseline_transform_list)

        print("Starting baseline experiment...")
        torch.manual_seed(42)
        np.random.seed(42)
        # start baseline experiment
        history = experimentation.cross_validate_model(
            model=setup.model,
            loss_fn=setup.loss_fn,
            data_loader_kwargs=setup.data_loader_kwargs,
            fit_kwargs=setup.fit_kwargs,
            fill_dict=setup.fill_dict,
            corrupt_func=setup.undo_min_max_corrupt_func,
            preprocess_init_fn=preprocess_init,
            device_ids=[dev],
        )

        np.save(os.path.join(cfg['experiment_directory'], 'mixed-transform-baseline.npy'), history)

    if args.optimal_loss_mix:
        transforms_list = [
            (group1, lambda : StandardScalerTimeSeries()),
            (group2, lambda : TanhStandardScalerTimeSeries()),
            (group3, lambda : StandardScalerTimeSeries()),
            (group4, lambda : StandardScalerTimeSeries()),
            (group5, lambda : TanhStandardScalerTimeSeries()),
            (group6, lambda : StandardScalerTimeSeries()),
            (group7, lambda : TanhStandardScalerTimeSeries()),
            (group8, lambda : TanhStandardScalerTimeSeries()),
            (group9, lambda : TanhStandardScalerTimeSeries()),
        ]

        preprocess_init = lambda : MixedTransformsTimeSeries(transforms_list)

        print("Starting experiment using optimal mixture from previous experiments...")
        torch.manual_seed(42)
        np.random.seed(42)
        # start baseline experiment
        history = experimentation.cross_validate_model(
            model=setup.model,
            loss_fn=setup.loss_fn,
            data_loader_kwargs=setup.data_loader_kwargs,
            fit_kwargs=setup.fit_kwargs,
            fill_dict=setup.fill_dict,
            corrupt_func=setup.undo_min_max_corrupt_func,
            preprocess_init_fn=preprocess_init,
            device_ids=[dev],
        )

        np.save(os.path.join(cfg['experiment_directory'], 'mixed-transform-optimal.npy'), history)


    for gid in group_ids:
        print(f"[{group_ids.index(gid) + 1} / {len(group_ids)}] Starting experiments for {gid} in {group_ids}")
        # For the specified group, try all alternative transformations
        transform_list = copy.deepcopy(baseline_transform_list)

        # just use default arguments to all of them
        for alt_transform, alt_name in zip([lambda : LogStandardScalerTimeSeries(), lambda : MinMaxTimeSeries(), lambda : LogMinMaxTimeSeries(), lambda : TanhStandardScalerTimeSeries()], ['log-standard', 'min-max', 'log-min-max', 'tanh-standard']):
            transform_list[gid] = (groups[gid], alt_transform)

            # do cross-validaiton and save
            print(f"Replacing group {gid} with {alt_name}")
            torch.manual_seed(42)
            np.random.seed(42)
            # start baseline experiment
            history = experimentation.cross_validate_model(
                model=setup.model,
                loss_fn=setup.loss_fn,
                data_loader_kwargs=setup.data_loader_kwargs,
                fit_kwargs=setup.fit_kwargs,
                fill_dict=setup.fill_dict,
                corrupt_func=setup.undo_min_max_corrupt_func,
                preprocess_init_fn=lambda : MixedTransformsTimeSeries(transform_list),
                device_ids=[dev],
            )

            np.save(os.path.join(cfg['experiment_directory'], f"mixed-transform-group-{gid}-{alt_name}.npy"), history)



