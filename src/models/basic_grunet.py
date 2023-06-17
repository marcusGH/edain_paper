import torch
import torch.nn as nn
import numpy as np

class GRUNetBasic(nn.Module):
    def __init__(self, num_features, hidden_dim, layer_dim, emb_dim, num_cat_columns = 11, dropout_prob = 0.2, dev = torch.device('cuda')):
        super(GRUNetBasic, self).__init__()

        # save the params
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.num_cat_columns = num_cat_columns
        self.dev = dev

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

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # First 11 columns are categorical, next 177 are numerical
        embedding_outs = []
        for k in range(self.num_cat_columns):
            emb = self.emb_layers[k]
            col = x[:, :, k].type(torch.int32)
            embedding_outs.append(emb(col))

        x = torch.concat([x[:, :, self.num_cat_columns:]] + embedding_outs, dim = -1)

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device = self.dev).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.sigmoid(self.fc3(out))

        return out.squeeze(1)