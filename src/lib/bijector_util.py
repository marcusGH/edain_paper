import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T

from tqdm.auto import tqdm
from src.lib import experimentation

def get_iaf_bijector(num_layers, scale_and_shift_dims, input_dim, dev, random_state=42):
    transforms = []
    layers_left = num_layers
    while layers_left > 0:
        # for reproducibility
        torch.random.manual_seed(random_state + layers_left)

        # create the permutations
        perm = torch.randperm(input_dim, dtype=torch.long, device=dev)
        inv_perm = torch.empty_like(perm, device=dev)
        inv_perm[perm] = torch.arange(input_dim, device=dev)

        # first affine autoregressive layer
        scale_and_shift = pyro.nn.AutoRegressiveNN(input_dim, scale_and_shift_dims, param_dims=[1, 1])
        transforms.append(T.AffineAutoregressive(scale_and_shift, stable=True))
        if layers_left == 1:
            break
        transforms.append(T.Permute(perm))

        # second affine autoregressive layer
        scale_and_shift = pyro.nn.AutoRegressiveNN(input_dim, scale_and_shift_dims, param_dims=[1, 1])
        transforms.append(T.AffineAutoregressive(scale_and_shift, stable=True))
        transforms.append(T.Permute(inv_perm))

        layers_left -= 2

    return T.ComposeTransformModule(transforms)

def fit_bijector(bijector, base_dist, train_loader, num_epochs=3, optimizer=None, scheduler=None, early_stopper=None, batch_preprocess_fn=None, inverse_fit=False, max_errors_ignore=20):
    """
    Set scheduler or early_stopper to "False" to disable them. Leaving as None uses default values
    """
    flow_dist = dist.TransformedDistribution(base_dist, [bijector])
    dev = next(bijector.parameters()).device

    if optimizer is None:
        optimizer = torch.optim.Adam(bijector.parameters(), lr=1e-3)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2, 3], gamma=0.1)
    if early_stopper is None:
        early_stopper = experimentation.EarlyStopper(patience=100)
    if batch_preprocess_fn is None:
        batch_preprocess_fn = lambda x : x

    stop = False

    for epoch_idx in tqdm(range(num_epochs), desc="Fitting bijector"):
        for X, _ in (pbar := tqdm(train_loader)):
            try:
                X = batch_preprocess_fn(X).to(dev)
                # compute loss and update gradients
                optimizer.zero_grad()
                if inverse_fit:
                    z = bijector.inv(X)
                    loss = -base_dist.log_prob(z).mean()
                else:
                    loss = -flow_dist.log_prob(X).mean()
                loss.backward()
                optimizer.step()
                # check for early stopping and report
                pbar.set_description(f"Bijector loss : {loss.detach().cpu().item():.4f}")
                if early_stopper and early_stopper.early_stop(loss.detach().cpu().item()):
                    stop = True
                    break
                # gc
                flow_dist.clear_cache()
                del X, loss
            except ValueError as e:
                warnings.warn("Skipping a batch due to value errors during training: " + str(e), RuntimeWarning)
                del X
                max_errors_ignore -= 1
                if max_errors_ignore <= 0:
                    raise e
                continue
        # at end of each epoch, step scheduler
        if scheduler:
            scheduler.step()
        if stop:
            break

def transform_data(bijector, data_loader, batch_preprocess_fn=None, batch_postprocess_fn=None) -> (np.ndarray, np.ndarray):
    """
    :param batch_preprocess_fn: function should map tensor of shape (N, D_1, ..., D_K) to rank-2 tensor of shape (N, D'), where all variables along the D' dimensions are numeric.
    :param batch_postprocess_fn: function should map rank 2 tensor (N, D') shape tensor to (N, D_1, ..., D_K) tensor
    """
    X_processed = []
    y_processed = []

    dev = next(bijector.parameters()).device

    if batch_preprocess_fn is None:
        batch_preprocess_fn = lambda x : x
    if batch_postprocess_fn is None:
        batch_postprocess_fn = lambda x : x
    dev = next(bijector.parameters()).device

    for X, y in tqdm(data_loader):
        with torch.no_grad():
            X_in = batch_preprocess_fn(X).to(dev)
            X_out = bijector.inv(X_in).detach().cpu()

            if not torch.all(torch.isfinite(X_out)):
                # find the culprits
                idx = torch.logical_not(torch.isfinite(X_out))
                idx = np.unique(np.where(idx)[0])
                warnings.warn(f"Replacing {len(idx)} erroneous sample(s) with other data during bijector data transformation.", RuntimeWarning)
                # then replace the erronous values and try again
                X_out[idx, :] = bijector.inv(prev_X_in[idx, :]).detach().cpu()

                # check if fixed after max number of iters or less
                if not torch.all(torch.isfinite(X_out)):
                    raise ValueError("Failed to remove non-finite values")
            # save working input to replace erroneous ones
            prev_X_in = X_in

        X = batch_postprocess_fn(X, X_out)
        # save
        X_processed.append(X)
        y_processed.append(y)
    return np.concatenate(X_processed, axis=0), np.concatenate(y_processed, axis=0)

def _validate_tensor(x, msg_prefix=""):
    if not torch.all(torch.isfinite(x)):
        num = torch.sum(~torch.isfinite(x))
        msg = f"There are {num} non-finite entries in tensor: {x}"
        raise ValueError(msg_prefix + msg)
