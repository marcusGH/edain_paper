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

from src.lib import experimentation
from src.models.basic_grunet import GRUNetBasic
import src.experiments.static_preprocessing_methods.experiment_setup as spm

def undo_min_max_corrupt_func(X, y):
    """
    X of shape (num_examples, series_length, num_features)
    In this undo, we assume scale same for each feature, over temporal scale
    """
    # to ensure we get the same mins and scales every time
    np.random.seed(42)
    # randomize both the starting point and the feature scales
    mins = np.random.uniform(-1E4, 1E4, size=X.shape[2])[np.newaxis, None]
    # don't set the smallest scale too tiny, otherwise can lose information due to float 32 bit
    scales = 10 ** np.random.uniform(-1, 5, size=X.shape[2])[np.newaxis, None]

    X_corrupt = X * scales + mins
    return X_corrupt, y

with open(os.path.join("config.yaml")) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

data_loader_kwargs = {
    'batch_size' : 1024,
    'shuffle' : False,
    'drop_last' : False,
}

fit_kwargs = {
        'train_split_data_dir' : os.path.join(cfg['dataset_directory'], "derived", "processed-splits"),
        'num_epochs' : 50,
        'early_stopper_patience' : 5,
        'early_stopper_min_delta' : 0.0,
        'optimizer_init' : lambda x: torch.optim.Adam(x, lr = 0.001),
        'scheduler_init' : lambda x: torch.optim.lr_scheduler.MultiStepLR(x, milestones=[4, 7], gamma=0.1),
        'verbose' : False,
}

fill_dict = {
    'nan' : -0.5,
    'pad_categorical' : -2,
    'pad_numeric' : -1.,
}

spec = importlib.util.spec_from_file_location("dain", os.path.join(cfg['dain_repo'], 'dain.py'))
dain = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = dain
spec.loader.exec_module(dain)

# mode='adaptive_avg', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144)  
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

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
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

        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.sigmoid(self.fc3(out))

        return out.squeeze(1)


torch.manual_seed(42)
np.random.seed(42)
# optimal learning rates for RNN according to paper
mean_lr, std_lr, scale_lr = 1e-02, 1e-8, 10
model = DainGRUNet(188, 128, 2, 4, mode='adaptive_scale', mean_lr=mean_lr, scale_lr=std_lr, gate_lr=scale_lr)
loss_fn = F.binary_cross_entropy
