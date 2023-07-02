import yaml
import os
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import importlib.util
import sys

from src.models.basic_grunet import GRUNetBasic

with open(os.path.join("config.yaml")) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

class AdaptiveGRUNet(nn.Module):
    def __init__(self, adaptive_layer, num_features, hidden_dim, layer_dim, emb_dim, num_cat_columns = 11, dropout_prob = 0.2, **adaptive_layer_kwargs):
        """
        This model takes input of shape (N, T, D)
        :param adaptive_layer: The class of the adaptive preporcessing layer. It's forward method
        should take a tensor of shape (N, D, T) and return tensor of same shape (not the axis swap)
        """
        super(AdaptiveGRUNet, self).__init__()

        # save the params
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.num_cat_columns = num_cat_columns

        # to get current device
        self.dummy_param = nn.Parameter(torch.empty(0))

        # the layers we need
        emb_layers = []
        for k in range(num_cat_columns):
            emb_layers.append(nn.Embedding(10, emb_dim))
        self.emb_layers = nn.ModuleList(emb_layers)

        self.gru = nn.GRU(
            input_size = num_features - num_cat_columns + num_cat_columns * emb_dim,
            hidden_size = hidden_dim,
            num_layers = layer_dim,
            batch_first = True,
            dropout = dropout_prob
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # DAIN preprocessing layer or other variant
        self.preprocess = adaptive_layer(input_dim=num_features - num_cat_columns, **adaptive_layer_kwargs)

    def forward(self, x):
        # First 11 columns are categorical, next 177 are numerical
        embedding_outs = []
        for k in range(self.num_cat_columns):
            emb = self.emb_layers[k]
            col = x[:, :, k].type(torch.int32)
            embedding_outs.append(emb(col))

        # apply DAIN preprocessing (or variant) just on the numeric columns
        preprocess_out = self.preprocess(x[:, :, self.num_cat_columns:].transpose(1, 2))

        x = torch.concat([preprocess_out.transpose(1, 2)] + embedding_outs, dim = -1)

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device = self.dummy_param.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        out = self.feed_forward(out)

        return out.squeeze(1)
