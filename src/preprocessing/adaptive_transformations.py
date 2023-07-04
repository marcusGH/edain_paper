import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from itertools import chain

class FullDAIN_Layer(nn.Module):
    def __init__(self, input_dim, time_series_length,
        adaptive_winsorization=True, adaptive_scale=True,
        adaptive_shift=True, adaptive_power_transform=True,
        eps=1e-8, dev=torch.device('cuda'),
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

        # for use in adaptive scale and shift
        self.running_mean = torch.zeros(self.D, device=dev)
        self.running_second_moment = torch.zeros(self.D, device=dev)
        self.running_n = torch.tensor(0, device=dev)

        # for adaptive winsorization
        self.alpha = nn.Parameter(
            data=torch.Tensor(1, self.T, self.D),
            requires_grad=True,
        )
        self.beta = nn.Parameter(
            data=torch.Tensor(1, self.T, self.D),
            requires_grad=True,
        )
        # default to tanh in (0, 1)
        nn.init.zeros_(self.beta)
        nn.init.normal_(self.alpha, std=0.2)

        # for adaptive shift
        self.shift_layer = nn.Linear(self.D, self.D, bias=False)
        self.shift_layer.weight.data = torch.FloatTensor(data=np.eye(self.D, self.D))
        self.skip_shift = nn.Parameter(
            data=torch.Tensor(1, self.T, self.D),
            requires_grad=True,
        )
        nn.init.constant_(self.skip_shift, 0.)

        # for adaptive scaling
        self.scaling_layer = nn.Linear(self.D, self.D, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(self.D, self.D))
        self.skip_scale = nn.Parameter(
            data=torch.Tensor(1, self.T, self.D),
            requires_grad=True,
        )
        nn.init.constant_(self.skip_scale, 0.)

        # for adaptive power-transform
        self.lambd = nn.Parameter(
            data=torch.Tensor(1, self.T, self.D),
            requires_grad=True,
        )
        self.skip_power_transform = nn.Parameter(
            data=torch.Tensor(1, self.T, self.D),
            requires_grad=True,
        )
        nn.init.ones_(self.lambd)
        nn.init.zeros_(self.skip_power_transform)

        # for saving power transform forward pass output
        self.power_transform_out = None # (N, T, D)
        def power_transform_backward_hook(_):
            """
            The default gradient computations for lambda update is very
            prone to numerical errors. Instead, we tune lambda by ignoring the
            default gradients, and instead considering the gradients from using
            a KL-divergence loss between the output samples and a normal distribution
            with the same mean and std.
            """
            x = self.power_transform_out # (N, T, D)
            mean_x = torch.mean(x, 0, keepdim=False)
            std_x = torch.std(x, 0, keepdim=False)
            # create a normal distribution with same moments as the input
            norm = torch.distributions.normal.Normal(loc=mean_x, scale=std_x)
            # Grads should be of shape (1, T, D)
            return torch.zeros_like(grad)
        self.lambd.register_hook(power_transform_backward_hook)

    def _adaptive_winsorization(self, x):
        """
        (1 - alpha') * x + alpha ' * beta' * tanh(x / beta')
         where alpha' in (0, 1) and beta' > 0
        """
        mean_x = torch.mean(x, 0, keepdim=True) # shape (T, D)
        # scale to mean 0, then scale back to ensure winsorization symmetric
        winsorized_x = torch.exp(self.beta) * torch.tanh((x - mean_x) * torch.exp(-self.beta)) + mean_x
        # residual bypass
        return torch.sigmoid(self.alpha) * x + (1. - torch.sigmoid(self.alpha)) * winsorized_x


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
        return chain([self.lambd, self.skip_power_transform])

    def scaling_params(self):
        return chain(self.scaling_layer.parameters(), [self.skip_scale])

    def shift_params(self):
        return chain(self.shift_layer.parameters(), [self.skip_shift])

    def forward(self, x, dim_first=True):
        """
        Input of shape (N, D, T) if dim_first=True, otherwise (N, T, D)
        Internally, we work with (N, T, D)
        """
        if dim_first:
            x = x.transpose(1, 2)

        # cumulative running estimates of E[X] and E[X^2], and sqrt(V[X])
        # this is done over both the temporal and n-axis, to get a result of shape (D,)
        assert (x.size(1), x.size(2)) == (self.T, self.D)
        mean_x = (self.running_mean * self.running_n + torch.sum(x.view((-1, self.D)), 0)) / (self.running_n + x.size(0) * x.size(1))
        mean_x2 = (self.running_second_moment * self.running_n + torch.sum(x.view((-1, self.D)) ** 2, 0)) / (self.running_n + x.size(0) * x.size(1))
        std = torch.sqrt(mean_x2 - mean_x ** 2 + self.eps)

        # Step 1: Winsorization
        if self.adaptive_winsorization:
            x = self._adaptive_winsorization(x)

        # Step 2: Shift
        if self.adaptive_shift:
            adaptive_avg = self.shift_layer(mean_x.unsqueeze(0)).unsqueeze(0) # (1, 1, D)
            x = (1. - torch.sigmoid(self.skip_shift)) * (x - adaptive_avg) + torch.sigmoid(self.skip_shift) * x

        # Step 3: Scale
        if self.adaptive_scale:
            adaptive_std = self.scaling_layer(std.unsqueeze(0)).unsqueeze(0) # (1, 1, D)
            # to avoid numerical errors
            adaptive_std[adaptive_std <= self.eps] = 1.0
            x = (1. - torch.sigmoid(self.skip_scale)) * (x / adaptive_std) + torch.sigmoid(self.skip_scale) * x

        # Step 4: Power transform
        if self.adaptive_power_transform:
            x = (1. - torch.sigmoid(self.skip_power_transform)) * self._adaptive_power_transform(x, self.lambd) + torch.sigmoid(self.skip_power_transform) * x

        # update running estimates
        self.running_mean = mean_x.detach()
        self.running_second_moment = mean_x2.detach()
        self.running_n += x.size(0) * x.size(1)

        # flip dimensions back
        if dim_first:
            x = x.transpose(1, 2)
        return x
