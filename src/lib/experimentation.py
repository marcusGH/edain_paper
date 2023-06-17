import torch
import torch.nn as nn
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
from tqdm.notebook import tqdm

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
        if validation_loss < (self.min_validation_loss + self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


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

def load_numpy_data(split_data_dir : str, val_idx : list, fill_dict, num_cats = 11, corrupt_func = None, preprocess_obj = None, **data_loader_kwargs) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
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
        for k in val_idx:
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
            Xs[:, :, num_cats:], y = corrupt_func(Xs[:, :, num_cats:], ys)

        # the transformation is only applied to the numeric columns
        if is_train and preprocess_obj is not None:
            Xs[:, :, num_cats:] = preprocess_obj.fit_transform(Xs[:, :, num_cats:], ys)
        elif preprocess_obj is not None:
            Xs[:, :, num_cats:] = preprocess_obj.transform(Xs[:, :, num_cats:])

        # compile the DataLoader object and return
        data_loader = torch.utils.data.DataLoader(
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(Xs).type(torch.float32),
                    torch.from_numpy(ys).type(torch.float32)
                    ), **data_loader_kwargs)

        return data_loader

    train_idx = [i for i in list(range(num_splits)) if i not in val_idx]
    train_loader = load_aux(train_idx, is_train = True)
    val_loader = load_aux(val_idx, is_train = False)
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


def fit_model(model, loss_fn, train_loader, val_loader, optimizer, scheduler = None, num_epochs = 10, verbose = True, dev = torch.device('cuda'), early_stopper = None):
    best_vloss = 1_000_000.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    history = {
            "timestamp" : timestamp,
            "train_loss" : [],
            "val_loss" : [],
            "train_amex_metric" : [],
            "val_amex_metric" : [],
            }

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

def cross_validate_model(model : nn.Module, loss_fn, data_loader_kwargs, fit_kwargs, fill_dict, corrupt_func, preprocess_init_fn,
                         folds = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]):
    """
    TODO
    """

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
        optimizer = fit_kwargs['optimizer_init'](model.parameters())
        if fit_kwargs['scheduler_init'] is not None:
            scheduler = fit_kwargs['scheduler_init'](optimizer)
        else:
            scheduler = None
        early_stopper = EarlyStopper(fit_kwargs['early_stopper_patience'], fit_kwargs['early_stopper_min_delta'])

        history = fit_model(model, loss_fn, train_loader, val_loader, optimizer, scheduler, fit_kwargs['num_epochs'], fit_kwargs['verbose'],
                early_stopper = early_stopper)

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
