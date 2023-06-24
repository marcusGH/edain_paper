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

spec = importlib.util.spec_from_file_location("dain", os.path.join(cfg['dain_repo'], 'dain.py'))
dain = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = dain
spec.loader.exec_module(dain)

class DainGRUNet(nn.Module):
    def __init__(self, num_features, hidden_dim, layer_dim, emb_dim, num_cat_columns = 11, dropout_prob = 0.2, **dain_kwargs):
        super(DainGRUNet, self).__init__()

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

        # DAIN preprocessing layer
        self.dain = dain.DAIN_Layer(input_dim=num_features - num_cat_columns, **dain_kwargs)

    def forward(self, x):
        # First 11 columns are categorical, next 177 are numerical
        embedding_outs = []
        for k in range(self.num_cat_columns):
            emb = self.emb_layers[k]
            col = x[:, :, k].type(torch.int32)
            embedding_outs.append(emb(col))

        # apply DAIN preprocessing just on the numeric columns
        dain_out = self.dain(x[:, :, self.num_cat_columns:].transpose(1, 2))

        x = torch.concat([dain_out.transpose(1, 2)] + embedding_outs, dim = -1)

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device = self.dummy_param.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        out = self.feed_forward(out)

        return out.squeeze(1)
