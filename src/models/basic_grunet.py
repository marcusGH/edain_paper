import torch
import torch.nn as nn
import numpy as np

class GRUNetBasic(nn.Module):
    def __init__(self, num_features, hidden_dim, layer_dim, embedding_dim, num_cat_columns = 11, dropout_prob = 0.2):
        super(GRUNetBasic, self).__init__()

        # save the params
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.num_cat_columns = num_cat_columns

        # to get current device
        self.dummy_param = nn.Parameter(torch.empty(0))

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
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device = self.dummy_param.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.sigmoid(self.fc3(out))

        return out.squeeze(1)


class GRUNetLOB(nn.Module):
    def __init__(self, input_dim=144, linear_dim=512, gru_dim=256, num_gru_layers=1, dropout_prob=0.5):
        super(GRUNetLOB, self).__init__()

        # save the params
        self.D = input_dim
        self.linear_dim = linear_dim
        self.gru_dim = gru_dim
        self.num_gru_layers = num_gru_layers

        # setup GRU RNN layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            # dropout=dropout_prob, # require more than 0 layers
        )

        # the classifier head. We're using cross-entropy with 3 classes, so output
        # should be unnormalized logits
        self.base = nn.Sequential(
            nn.Linear(gru_dim, linear_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(linear_dim, 3),
        )

    def forward(self, x):

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_gru_layers, x.size(0), self.gru_dim, device=x.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        x, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        x = x[:, -1, :]

        # pass through classifier head
        x = self.base(x)

        return x
