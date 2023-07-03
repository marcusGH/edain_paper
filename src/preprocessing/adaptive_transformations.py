import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from itertools import chain

class FullDAIN_Layer(nn.Module):
    def __init__(self, input_dim, time_series_length,
        adaptive_winsorization=True, adaptive_scale=True,
        adaptive_shift=True, adaptive_power_transform=True,
        eps=1e-8,
    ):
        super(FullDAIN_Layer, self).__init__()

        # for getting the current device
        self.dummy_param = nn.Parameter(torch.empty(0))

        self.D = input_dim
        self.T = time_series_length
        self.eps = eps

        self.adaptive_winsorization = adaptive_winsorization
        self.adaptive_scale = adaptive_scale
        self.adaptive_shift = adaptive_shift
        self.adaptive_power_transform = adaptive_power_transform

        # for adaptive winsorization
        self.alpha = nn.Parameter(
            data=torch.Tensor(1, self.D, self.T),
            requires_grad=True,
        )
        self.beta = nn.Parameter(
            data=torch.Tensor(1, self.D, self.T),
            requires_grad=True,
        )
        # default to tanh in (0, 1)
        nn.init.zeros_(self.beta)
        nn.init.normal_(self.alpha, std=0.2)

        # for adaptive shift
        self.shift_layer = nn.Linear(self.D, self.D, bias=False)
        self.shift_layer.weight.data = torch.FloatTensor(data=np.eye(self.D, self.D))

        # for adaptive scaling
        self.scaling_layer = nn.Linear(self.D, self.D, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(self.D, self.D))

        # for adaptive power-transform
        self.lambd = nn.Parameter(
            data=torch.Tensor(1, self.D, self.T),
            requires_grad=True,
        )
        nn.init.ones_(self.lambd)

        # for saving power transform forward pass output
        self.power_transform_out = None # (N, D, T)
        def power_transform_backward_hook(_):
            """
            The default gradient computations for lambda update is very
            prone to numerical errors. Instead, we tune lambda by ignoring the
            default gradients, and instead considering the gradients from using
            a KL-divergence loss between the output samples and a normal distribution
            with the same mean and std.
            """
            x = self.power_transform_out # (N, D, T)
            mean_x = torch.mean(x, 0, keepdim=False)
            std_x = torch.std(x, 0, keepdim=False)
            # create a normal distribution with same moments as the input
            norm = torch.distributions.normal.Normal(loc=mean_x, scale=std_x)
            # Grads should be of shape (1, D, T)
            return torch.zeros_like(grad)
        self.lambd.register_hook(power_transform_backward_hook)

    def _adaptive_winsorization(self, x):
        """
        (1 - alpha') * x + alpha ' * beta' * tanh(x / beta')
         where alpha' in (0, 1) and beta' > 0
        """
        mean_x = torch.mean(x, 0, keepdim=True) # shape (D, T)
        # scale to mean 0, then scale back to ensure winsorization symmetric
        winsorized_x = torch.exp(self.beta) * torch.tanh((x - mean_x) * torch.exp(-self.beta)) + mean_x
        # residual bypass
        return (1 - torch.sigmoid(self.alpha)) * x + torch.sigmoid(self.alpha) * winsorized_x

    def _adaptive_shift(self, x):
        # Input of shape (N, D, T)
        avg = torch.mean(x, 2) # (N, D)
        adaptive_avg = self.shift_layer(avg) # (N, D)
        return x - adaptive_avg.unsqueeze(-1)

    def _adaptive_scale(self, x):
        # Input of shape (N, D, T)
        std = torch.mean(x ** 2, 2) # (N, D)
        std = torch.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std)
        # to avoid numerical errors
        adaptive_std[adaptive_std <= self.eps] = 1
        return x / adaptive_std.unsqueeze(-1)

    def _adaptive_power_transform(self, x, lambd):
        """
        References:
          - Yeo and Johnson (2000)
        """
        # Case 0: lambd not 0, y non-negative
        idx = (torch.abs(lambd) > self.eps) & (x >= 0.0)
        x[idx] = ((torch.pow(x + 1, lambd) - 1) / lambd)[idx]

        # Case 1: lambd = 0, y non-negative
        idx = (torch.abs(lambd) <= self.eps) & (x >= 0.0)
        x[idx] = torch.log1p(x[idx])

        # Case 2: lambd not 2, y negative
        idx = (torch.abs(lambd - 2) > self.eps) & (x < 0.0)
        x[idx] = ((torch.pow(1 - x, 2 - lambd) - 1) / (lambd - 2))[idx]

        # Case 3: lambd = 2, y negative
        idx = (torch.abs(lambd - 2) <= self.eps) & (x < 0.0)
        x[idx] = -torch.log1p(-x[idx])

        # required for computing backward gradients
        self.power_transform_out = x
        return x

    def winsorization_params(self):
        # https://stackoverflow.com/questions/69774137/constructing-parameter-groups-in-pytorch
        return chain([self.alpha, self.beta])

    def power_transform_params(self):
        # https://stackoverflow.com/questions/69774137/constructing-parameter-groups-in-pytorch
        return chain([self.lambd])

    def forward(self, x, dim_first=True):
        """
        Input of shape (N, D, T) if dim_first=True, otherwise (N, T, D)
        """
        if not dim_first:
            x = x.transpose(1, 2)

        # Step 1: Winsorization
        if self.adaptive_winsorization:
            x = self._adaptive_winsorization(x)

        # Step 2: Shift
        if self.adaptive_shift:
            x = self._adaptive_shift(x)

        # Step 3: Scale
        if self.adaptive_scale:
            x = self._adaptive_scale(x)

        # Step 4: Power transform
        if self.adaptive_power_transform:
            x = self._adaptive_power_transform(x, self.lambd)

        if not dim_first:
            x = x.tranpose(1, 2)
        return x
