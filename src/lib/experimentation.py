import torch
import time
import torch.nn as nn
import warnings
import torch.nn.functional as F
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc
import cupy
import cudf
import sklearn

from datetime import datetime
from tqdm.auto import tqdm
from src.models import basic_grunet
from src.lib.lob_train_utils import lob_epoch_train_one_epoch, lob_evaluator
from src.lib.lob_loader import get_wf_lob_loaders, ImbalancedDatasetSampler

class EarlyStopper:
    """
    References:
      - https://stackoverflow.com/a/73704579
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# undo the min-max preprocessing
def undo_min_max_corrupt_func(X, y, random_state=42):
    """
    X of shape (num_examples, series_length, num_features)
    In this undo, we assume scale same for each feature, over temporal scale
    """
    # to ensure we get the same mins and scales every time
    np.random.seed(random_state)
    # randomize both the starting point and the feature scales
    mins = np.random.uniform(-1E4, 1E4, size=X.shape[2])[np.newaxis, None]
    # don't set the smallest scale too tiny, otherwise can lose information due to float 32 bit
    scales = 10 ** np.random.uniform(-1, 5, size=X.shape[2])[np.newaxis, None]

    X_corrupt = X * scales + mins
    return X_corrupt, y

def amex_metric_mod(y_true, y_pred):
    """
    COMPETITION METRIC FROM Konstantin Yakovlev

    References:
      - https://www.kaggle.com/kyakovlev
      - https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
    """
    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

def load_amex_numpy_data(split_data_dir, fill_dict, corrupt_func=None, num_categorical_features=11, load_small_subset=False):
    """
    :param split_data_dir: should be the directory where the train data and targets are located.
    :param fill_dict: should contain the following keys:
        * nan
        * pad_categorical
        * pad_numeric
    :param corrupt_func: should be a function that takes in a numpy
          array and returns a corrupted version of it of same shape
    :param num_categorical_features: number of categorical features in the dataset

    :returns: (X, y) numpy.ndarrays of the data and targets
    """
    data_files = [name for name in os.listdir(split_data_dir) if os.path.isfile(os.path.join(split_data_dir, name))]

    Xs = []; ys = []
    for k in range(len(data_files) // 2):
        Xs.append(np.load(os.path.join(split_data_dir, f"train-data_{k}.npy")))
        ys.append(pd.read_parquet(os.path.join(split_data_dir, f"train-targets_{k}.parquet")))
        # for testing, more efficient to just load one of the files
        if load_small_subset:
            break

    Xs = np.concatenate(Xs, axis = 0)
    ys = pd.concat(ys).target.values

    # fill NAs and padded values with provided numerics
    # (See PAD_CUSTOMER_TO_13_ROWS code)
    na_mask = (Xs == -0.5)
    pad_cat_mask = (Xs == -2)
    pad_numeric_mask = (Xs == -3)

    Xs[na_mask] = fill_dict['nan']
    Xs[pad_cat_mask] = fill_dict['pad_categorical']
    Xs[pad_numeric_mask] = fill_dict['pad_numeric']

    # make sure all the categorical entries are non-negative for the embedding layer to work correctly
    if num_categorical_features is not None:
        Xs[:, :, :num_categorical_features] = \
            Xs[:, :, :num_categorical_features] - \
            np.amin(Xs[:, :, :num_categorical_features], axis=0, keepdims=True)

    # corrupt the data with specified function (only do this on the non-categorical variables)
    if corrupt_func is not None:
        Xs[:, :, num_categorical_features:], ys = corrupt_func(Xs[:, :, num_categorical_features:], ys)

    return Xs, ys

def load_numpy_data(split_data_dir : str, val_idx : list, fill_dict, num_cats = 11, corrupt_func = None, preprocess_obj = None, dtype=torch.float32, **data_loader_kwargs) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """
    val_idx is a list of integers denoting which of the [0, 1, ..., NUM_SPLITS-1] splits to use
    as validation data, and the rest will be used as training data

    if num_cat_columns is set to an integer, the first num_cat_columns columns will have their values shifted to all be non-negative,
    as these are categorical integer indices.

    :param split_data_dir: should be the directory where the train data and targets are located. The number of splits should be the same
    for both train data and targets

    The fill dict should contain the following keys:
    * nan
    * pad_categorical
    * pad_numeric
    """
    data_files = [name for name in os.listdir(split_data_dir) if os.path.isfile(os.path.join(split_data_dir, name))]
    num_splits = len(data_files) // 2

    def load_aux(idx : list, is_train : bool):
        Xs = []; ys = []
        for k in idx:
            Xs.append(np.load(os.path.join(split_data_dir, f"train-data_{k}.npy")))
            ys.append(pd.read_parquet(os.path.join(split_data_dir, f"train-targets_{k}.parquet")))

        Xs = np.concatenate(Xs, axis = 0)
        ys = pd.concat(ys).target.values

        # fill NAs and padded values with provided numerics
        # (See PAD_CUSTOMER_TO_13_ROWS code)
        na_mask = (Xs == -0.5)
        pad_cat_mask = (Xs == -2)
        pad_numeric_mask = (Xs == -3)

        Xs[na_mask] = fill_dict['nan']
        Xs[pad_cat_mask] = fill_dict['pad_categorical']
        Xs[pad_numeric_mask] = fill_dict['pad_numeric']

        # make sure all the categorical entries are non-negative for the embedding layer to work correctly
        if num_cats is not None:
            Xs[:, :, :num_cats] = Xs[:, :, :num_cats] - np.amin(Xs[:, :, :num_cats], axis = 0, keepdims = True)

        # corrupt the data with specified function (only do this on the non-categorical variables)
        if corrupt_func is not None:
            Xs[:, :, num_cats:], ys = corrupt_func(Xs[:, :, num_cats:], ys)

        # the transformation is only applied to the numeric columns
        if is_train and preprocess_obj is not None:
            Xs[:, :, num_cats:] = preprocess_obj.fit_transform(Xs[:, :, num_cats:], ys)
        elif preprocess_obj is not None:
            Xs[:, :, num_cats:] = preprocess_obj.transform(Xs[:, :, num_cats:])

        # compile the DataLoader object and return
        data_loader = torch.utils.data.DataLoader(
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(Xs).type(dtype),
                    torch.from_numpy(ys).type(dtype)
                    ), **data_loader_kwargs)

        return data_loader

    train_idx = [i for i in list(range(num_splits)) if i not in val_idx]
    train_loader = load_aux(train_idx, is_train = True) if len(train_idx) > 0 else None
    val_loader = load_aux(val_idx, is_train = False) if len(val_idx) > 0 else None
    return train_loader, val_loader

def train_one_epoch(model, loss_fn, training_loader, optimizer, epoch_number, dev=torch.device('cuda')):
    running_loss = 0.
    running_metric = 0.
    # last_loss = 0.
    # last_metric = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(dev), labels.to(dev)

        # sanity check
        if not torch.all(torch.isfinite(inputs)):
            msg = f"Encountered {len(inputs[~torch.isfinite(inputs)])} non-finite input values at iteration {i} during training"
            raise ValueError(msg)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        running_metric += amex_metric_mod(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())

    last_loss = running_loss / (i + 1) # loss per batch
    last_metric = running_metric / (i + 1) # metric per batch
    # print('  batch {} loss: {} metric: {}'.format(i + 1, last_loss, last_metric))
    # tb_x = epoch_index * len(training_loader) + i + 1
    # tb_writer.add_scalar('Loss/train', last_loss, tb_x)

    return last_loss, last_metric


def fit_model(model, loss_fn, train_loader, val_loader, optimizer, scheduler = None, num_epochs = 10, verbose = True, early_stopper = None, device_ids = None):
    best_vloss = 1_000_000.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    history = {
        "timestamp" : timestamp,
        "train_loss" : [],
        "val_loss" : [],
        "train_amex_metric" : [],
        "val_amex_metric" : [],
    }

    if isinstance(device_ids, torch.device):
        dev = device_ids
    elif device_ids == "cpu":
        dev = torch.device('cpu')
    elif device_ids is not None and len(device_ids) > 1:
        # create parallel model and move it to the main gpu
        model = nn.DataParallel(model, device_ids = device_ids)
        dev = torch.device('cuda', device_ids[0])
    elif device_ids is not None:
        dev = torch.device('cuda', device_ids[0])
    else:
        dev = torch.device('cuda')
    print(f"Using device = {dev}")
    model = model.to(dev)

    pbar = tqdm(total = num_epochs)

    for epoch in range(num_epochs):
        # print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, avg_metric = train_one_epoch(model, loss_fn, train_loader, optimizer, epoch + 1, dev = dev)

        running_vloss = 0.0
        running_vmetric = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(dev), vlabels.to(dev)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels).cpu().item()
                vmetric = amex_metric_mod(vlabels.cpu().numpy(), voutputs.cpu().numpy())
                running_vloss += vloss
                running_vmetric += vmetric

        avg_vloss = running_vloss / (i + 1)
        avg_vmetric = running_vmetric / (i + 1)
        if verbose:
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            print('AMEX metric train {} valid {}'.format(avg_metric, avg_vmetric))

        # update learning rate
        if scheduler is not None:
            scheduler.step()

        # update progress bar
        pbar.update()
        pbar.set_description("LOSS train {:.4f} valid {:.4f}. AMEX METRIC train {:.4f} valid {:.4f}".format(avg_loss, avg_vloss, avg_metric, avg_vmetric))

        # Log the metric values
        history['train_loss'].append(avg_loss)
        history['train_amex_metric'].append(avg_metric)
        history['val_loss'].append(avg_vloss)
        history['val_amex_metric'].append(avg_vmetric)

        # Log the running loss averaged per batch
        # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                 epoch_number + 1)
        # writer.flush()

        # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        #     model_path = 'model_{}_{}'.format(timestamp, epoch)
        #     torch.save(model.state_dict(), model_path)

        # early stopper
        if early_stopper is not None and early_stopper.early_stop(avg_vloss):
            break
    pbar.refresh()

    return history

def cross_validate_experiment(
        model_init_fn,
        preprocess_init_fn,
        optimizer_init_fn,
        scheduler_init_fn,
        early_stopper_init_fn,
        loss_fn,
        X,
        y,
        num_epochs,
        dataloader_kwargs,
        num_folds,
        device,
        random_state,
        num_categorical_features,
    ):
    """
    Cross-validates the provided model on provided data with provided preprocessing method

    :param model_init_fn: function that returns a model
    :param preprocess_init_fn: function that returns a preprocessing method
    :param loss_fn: loss function
    :param X: np.ndarray of shape (n_samples, n_timesteps, n_features)
    :param y: np.ndarray of labels of shape (n_samples,)
    :param num_folds: number of folds to use
    :param device: device to use for training
    :param random_state: random state to use for splitting data
    """
    history_metrics = {
        "train_loss": [None] * num_folds,
        "val_loss": [None] * num_folds,
        "train_amex_metric": [None] * num_folds,
        "val_amex_metric": [None] * num_folds,
    }
    history_num_epochs = [None] * num_folds
    history_preprocess_time = [None] * num_folds
    history_train_time = [None] * num_folds

    # split data into folds
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if num_folds > 1:
        kf = sklearn.model_selection.KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    else:
        # if only 1 fold, we instead do 80%-20% split and only train once
        kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=random_state)
    for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Starting model training [{i+1} / {num_folds}]")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # preprocess data
        start_time = time.time()
        preprocess = preprocess_init_fn()
        X_train[:, :, num_categorical_features:] = preprocess.fit_transform(X_train[:, :, num_categorical_features:], y_train)
        X_val[:, :, num_categorical_features:] = preprocess.transform(X_val[:, :, num_categorical_features:])
        preprocess_time = time.time() - start_time

        # create the data loaders
        train_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).float(),
            ),
            **dataloader_kwargs
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).float(),
            ),
            **dataloader_kwargs
        )

        # initialize model as well as optimizer, scheduler, and early stopper
        model = model_init_fn()
        optimizer = optimizer_init_fn(model)
        scheduler = scheduler_init_fn(optimizer)
        early_stopper = early_stopper_init_fn()

        start_time = time.time()
        history = fit_model(
            model, loss_fn,
            train_loader, val_loader,
            optimizer, scheduler,
            num_epochs, verbose=False,
            early_stopper=early_stopper, device_ids=device
        )
        train_time = time.time() - start_time

        # save metrics
        for history_key in history_metrics.keys():
            history_metrics[history_key][i] = np.array(history[history_key])
        history_num_epochs[i] = len(history['train_loss'])
        history_preprocess_time[i] = preprocess_time
        history_train_time[i] = train_time

        if num_folds == 1:
            break

    hist_keys = list(history_metrics.keys())
    for history_key in hist_keys:
        # compute the mean and std of the final value, reducing over folds
        final_values = [history_metrics[history_key][i][-1] for i in range(num_folds)]
        history_metrics[f"{history_key}_mean"] = np.mean(final_values)
        history_metrics[f"{history_key}_sd"] = np.std(final_values)
    history_metrics['num_epochs'] = history_num_epochs
    history_metrics['preprocess_times'] = history_preprocess_time
    history_metrics['train_times'] = history_train_time
    return history_metrics

def cross_validate_model(model : nn.Module, loss_fn, data_loader_kwargs, fit_kwargs, fill_dict, corrupt_func, preprocess_init_fn,
                         folds = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
                         device_ids = None):
    """
    TODO

    This method is depracated. Use cross_validate_experiment instead.
    """
    w = DeprecationWarning("This method is depracated. Use cross_validate_experiment instead.")
    warnings.warn(w)


    history_metrics = {
            "train_loss"        : [None] * len(folds),
            "val_loss"          : [None] * len(folds),
            "train_amex_metric" : [None] * len(folds),
            "val_amex_metric"   : [None] * len(folds),
            }
    num_epochs = [None] * len(folds)

    for i, val_idx in tqdm(enumerate(folds), desc = "Cross-validating model", total = len(folds)):
        # we want to train the model from scratch
        reset_all_weights(model)

        train_loader, val_loader = load_numpy_data(fit_kwargs['train_split_data_dir'], val_idx,
                fill_dict=fill_dict,
                corrupt_func=corrupt_func,
                preprocess_obj=preprocess_init_fn(),
                **data_loader_kwargs)

        # initialise the optimizer and learning rate scheduler, and an early stopper
        optimizer = fit_kwargs['optimizer_init'](model)
        if fit_kwargs['scheduler_init'] is not None:
            scheduler = fit_kwargs['scheduler_init'](optimizer)
        else:
            scheduler = None
        early_stopper = EarlyStopper(fit_kwargs['early_stopper_patience'], fit_kwargs['early_stopper_min_delta'])

        history = fit_model(model, loss_fn, train_loader, val_loader, optimizer, scheduler, fit_kwargs['num_epochs'], fit_kwargs['verbose'],
                early_stopper = early_stopper, device_ids = device_ids)

        # save the various metrics recorded
        for history_key in history_metrics.keys():
            history_metrics[history_key][i] = np.array(history[history_key])
        num_epochs[i] = len(history['train_loss'])

    hist_keys = list(history_metrics.keys())
    for history_key in hist_keys:
        # compute the mean and std of the final value, reducing over folds
        final_values = [history_metrics[history_key][i][-1] for i in range(len(folds))]
        history_metrics[f"{history_key}_mean"] = np.mean(final_values)
        history_metrics[f"{history_key}_sd"] = np.std(final_values)
    history_metrics['num_epochs'] = num_epochs

    print("Done!")

    return history_metrics

def reset_all_weights(model: nn.Module) -> None:
    """
    References:
      - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
      - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
      - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


def train_evaluate_lob_anchored(
        h5_file_path,
        model_init_fn,
        preprocess_init_fn,
        optimizer_init_fn,
        scheduler_init_fn,
        early_stopper_init_fn,
        num_epochs,
        device,
        random_state,
        horizon=2,
        windows=15,
        batch_size=128,
        use_resampling=True,
        splits=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    ):
    """
    TODO: docstring
    """

    # to avoid errors with the data loader for opening to many files
    torch.multiprocessing.set_sharing_strategy('file_system')

    history = {
        "split_results" : [],
        "splits" : splits,
        "train_time" : [],
    }

    for i in splits:
        print(f"#### Evaluating model for split {i} ####")
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        # get the train and test data loaders
        train_loader, val_loader = get_wf_lob_loaders(
            h5_path=h5_file_path,
            window=windows,
            horizon=horizon,
            split=i,
            batch_size=batch_size,
            class_resample=use_resampling,
            normalization=None,
        )

        ###### Fitting the preprocessing object on train data ######

        if preprocess_init_fn is not None:
            print(f"Fitting preprocesser to data for split {i}")
            preprocess = preprocess_init_fn()

            # turn train loader into numpy array
            X_train, y_train = [], []
            for X, y in train_loader:
                X_train.append(X.numpy())
                y_train.append(y.numpy())
            X_train = np.concatenate(X_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)

            # fit the preprocessing object
            preprocess.fit(X_train, y_train)
        else:
            preprocess = None

        ######        Setup the model, optimizer, etc.        ######

        # setup mode
        model = model_init_fn()
        model.to(device)
        # setup optimizer and scheduler
        model_optimizer = optimizer_init_fn(model)
        model_scheduler = scheduler_init_fn(model_optimizer)
        # setup early stopper
        early_stopper = early_stopper_init_fn()


        ######               Start training loop              ######

        results = []

        start_time = time.time()
        for epoch in (pbar := tqdm(range(num_epochs))):
            # train one epoch
            train_loss = lob_epoch_train_one_epoch(model, train_loader, preprocess, model_optimizer, device)

            # evaluate on validation set, and save metrics
            metrics = lob_evaluator(model, val_loader, preprocess, device)
            metrics['train_loss'] = train_loss
            results.append(metrics)

            # update progress bar
            pbar.set_description(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Val loss: {metrics['val_loss']:.4f} | Val kappa: {metrics['kappa']:.4f}")

            # update scheduler
            if model_scheduler is not None:
                model_scheduler.step()

            # check early stopper (not high kappa is good, so invert)
            if early_stopper is not None and early_stopper.early_stop(metrics['val_loss']):
                break
        pbar.refresh()
        pbar.close()

        train_time = time.time() - start_time
        history['split_results'].append(results)
        history['train_time'].append(train_time)

    return history
