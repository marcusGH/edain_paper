import yaml
import os
import torch
import torch.nn as nn

class AdaptiveGRUNet(nn.Module):
    def __init__(self, adaptive_layer, num_features, hidden_dim, layer_dim, embedding_dim, num_cat_columns=11, time_series_length=13, dropout_prob=0.2, dim_first=True):
        """
        This model takes input of shape (N, T, D). It returns probabilities of shape (N,)

        :param adaptive_layer: a pytorch module that takes a tensor of
        shape (N, D', T) if dim_first=True, otherwise (N, T, D') and returns a tensor of same shape,
        where D' = D - num_cat_columns
        """
        super(AdaptiveGRUNet, self).__init__()

        # save the params
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.num_cat_columns = num_cat_columns

        # the layers we need
        emb_layers = []
        for k in range(num_cat_columns):
            emb_layers.append(nn.Embedding(10, embedding_dim))
        self.emb_layers = nn.ModuleList(emb_layers)

        self.gru = nn.GRU(
            input_size =num_features - num_cat_columns + num_cat_columns * embedding_dim,
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

        # adaptive preprocessing layer
        self.D = num_features - num_cat_columns
        self.T = time_series_length
        self.dim_first = dim_first
        self.preprocess = adaptive_layer

    def forward(self, x):
        # First 11 columns are categorical, next 177 are numerical
        embedding_outs = []
        for k in range(self.num_cat_columns):
            emb = self.emb_layers[k]
            col = x[:, :, k].type(torch.int32)
            embedding_outs.append(emb(col))

        # apply adaptive preprocessing just on the numeric columns
        if self.dim_first:
            # reshape to (N, D', T), then transpose back to (N, T, D')
            preprocess_out = self.preprocess(x[:, :, self.num_cat_columns:].transpose(1, 2)).transpose(1, 2)
        else:
            preprocess_out = self.preprocess(x[:, :, self.num_cat_columns:])

        x = torch.concat([preprocess_out] + embedding_outs, dim=-1)

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        out = self.feed_forward(out)
        return out.squeeze(1)

class AdaptiveGRUNetLOB(nn.Module):
    def __init__(self, adaptive_layer, time_series_length, dim_first, input_dim=144, linear_dim=512, gru_dim=256, num_gru_layers=1, dropout_prob=0.5):
        super(AdaptiveGRUNetLOB, self).__init__()

        # save the params
        self.D = input_dim
        self.T = time_series_length
        self.dim_first = dim_first
        self.linear_dim = linear_dim
        self.gru_dim = gru_dim
        self.num_gru_layers = num_gru_layers

        # the adaptive preprocessing layer
        self.preprocess = adaptive_layer

        # setup GRU RNN layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            # dropout=dropout_prob, # require more than 0 layers
        )

        # dummy parameter
        self.emb_layers = nn.Parameter(torch.empty(0))

        # the classifier head. We're using cross-entropy with 3 classes, so output
        # should be unnormalized logits
        self.feed_forward = nn.Sequential(
            nn.Linear(gru_dim, linear_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(linear_dim, 3),
        )

    def forward(self, x):
        # apply adaptive preprocessing just on the numeric columns
        if self.dim_first:
            # reshape to (N, D', T), then transpose back to (N, T, D')
            x = self.preprocess(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.preprocess(x)

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_gru_layers, x.size(0), self.gru_dim, device=x.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        x, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        x = x[:, -1, :]

        # pass through classifier head
        x = self.feed_forward(x)

        return x
