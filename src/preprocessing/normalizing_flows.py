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
from src.lib.bijector_util import _validate_tensor

class AdaptiveScale(dist.torch_transform.TransformModule):
    """
    Todo
    """

    # setup params
    domain = dist.transforms.constraints.real_vector # or just real?
    codomain = dist.transforms.constraints.real_vector
    bijective = True

    def __init__(self, input_dim, init_sigma=0.1):
        super(AdaptiveScale, self).__init__(cache_size=1)

        self.input_dim = input_dim
        # learned standard deviation should be positive
        self.log_scale = nn.Parameter(
            torch.randn(self.input_dim) * init_sigma
        )

    def _params(self):
        # s = torch.exp(self.log_scale)
        return self.log_scale

    def _call(self, x):
        # apply scale
        return x * torch.exp(self.log_scale)

    def _inverse(self, y):
        x = y * torch.exp(-self.log_scale)
        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian
        """
        return torch.sum(self.log_scale)

    def _inverse_log_abs_det_jacobian(self, x, y):
        return torch.sum(-self.log_scale)


class AdaptiveShift(dist.torch_transform.TransformModule):
    """
    Todo
    """

    # setup params
    domain = dist.transforms.constraints.real_vector # or just real?
    codomain = dist.transforms.constraints.real_vector
    bijective = True

    def __init__(self, input_dim, init_sigma=0.1):
        super(AdaptiveShift, self).__init__(cache_size=1)

        self.input_dim = input_dim
        # learned shift can be any real
        self.shift = nn.Parameter(
            torch.randn(self.input_dim) * init_sigma
        )

    def _params(self):
        # s = torch.exp(self.log_scale)
        return self.shift

    def _call(self, x):
        return x + self.shift

    def _inverse(self, y):
        return y - self.shift

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian
        """
        return torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

    def _inverse_log_abs_det_jacobian(self, x, y):
        return torch.zeros(y.size(0), device=y.device, dtype=y.dtype)


class AdaptiveOutlierRemoval(dist.torch_transform.TransformModule):
    """
    Only invertible if residual_connection is set to False

    Performs the transformation:

       y = (1-a') * x + a' * b' * tanh(x / b')

    where a' in [0, 1] and b' > 0
    """

    # setup params
    domain = dist.transforms.constraints.real_vector # or just real?
    codomain = dist.transforms.constraints.real_vector
    bijective = True

    def __init__(self, input_dim, init_sigma=0.1, residual_connection=True, mode='exp'):
        super(AdaptiveOutlierRemoval, self).__init__(cache_size=1)

        self.input_dim = input_dim
        # learned b' in above equation should be positive, which we either constrain with softplus
        # or an exponential function, depending on the mode
        assert mode in ['exp', 'softplus']
        self.mode = mode
        self.log_cutoff = nn.Parameter(
            torch.randn(self.input_dim) * init_sigma
        )

        # skip-parameter, before applying sigmoid
        self.alpha = None
        if residual_connection:
            self.alpha = nn.Parameter(
                torch.randn(self.input_dim) * init_sigma
            )

    def _params(self):
        if self.alpha is None:
            return self.log_cutoff
        else:
            return self.log_cutoff, self.alpha

    def _call(self, x):
        """
        If residual connection:
          (1-a') * x + a' * (b' * tanh(x / b'))
        Otherwise:
          b' * tanh(x / b')
        """
        _validate_tensor(x, "Outlier removal input: ")

        if self.mode == 'exp':
            beta = torch.exp(self.log_cutoff)
        elif self.mode == 'softplus':
            beta = F.softplus(self.self.log_cutoff)
        else:
            raise NotImplementedError("Invalid mode: " + self.mode)

        x_tanh = beta * torch.tanh(x / beta)
        if self.alpha is not None:
            y = (1. - torch.sigmoid(self.alpha)) * x + torch.sigmoid(self.alpha) * x_tanh
        else:
            y = x_tanh

        _validate_tensor(y, "Outlier removal forwards: ")

        return y

    def _inverse(self, y):
        if self.alpha is not None:
            raise NotImplementedError("There is not analytical expression for inverting adaptive outlier removal transformation when using a residual connection")

        if self.mode == 'exp':
            beta = torch.exp(self.log_cutoff)
        elif self.mode == 'softplus':
            beta = F.softplus(self.self.log_cutoff)
        else:
            raise NotImplementedError("Invalid mode: " + self.mode)

        return torch.atanh(y / beta) * beta

    def log_abs_det_jacobian(self, x, y):
        """
        References:
          - https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/bijectors/tanh.py#L72-L83

        The below formula is equivalent to `torch.log1p(-torch.square(torch.tanh(x / b')))`, but is more numerically
        stable. See references for proof.
        """
        if self.alpha is not None:
            raise NotImplementedError("There is not analytical expression for log abs det jacobian for the adaptive outlier removal transformation when using a residual connection")

        if self.mode == 'exp':
            beta = torch.exp(self.log_cutoff)
        elif self.mode == 'softplus':
            beta = F.softplus(self.self.log_cutoff)
        else:
            raise NotImplementedError("Invalid mode: " + self.mode)

        ladj = 2. * (np.log(2.) - x / beta - F.softplus(-2. * x / beta))
        # determinant of diagonal matrix is just sum since we're taking logs
        return torch.sum(ladj, axis=-1)

    def _inverse_log_abs_det_jacobian(self, x, y):
        """
        This is a more numerically stable version of `-torch.log1p(-torch.square(y * b'))`, but still not that stable.
        If possible, try to use the forward log abs det jacobian instead.
        """
        if self.alpha is not None:
            raise NotImplementedError("There is not analytical expression for inverse log abs det jacobian for the adaptive outlier removal transformation when using a residual connection")

        if self.mode == 'exp':
            beta = torch.exp(self.log_cutoff)
        elif self.mode == 'softplus':
            beta = F.softplus(self.self.log_cutoff)
        else:
            raise NotImplementedError("Invalid mode: " + self.mode)

        #  Derivation:
        # ----------------------------------------
        #    -log(|1-y'^2|)
        #  = -log(|1-y'| * |1+y'|)
        #  = -log(|1 - y'|) - log(|1 + y'|)
        #  =    (part 1)   +    (part 2)
        #
        #  Part 1:
        #    -log(|1 - y'|)
        #  = (case 1: 1 < y')  -log(y' - 1)
        #    (case 2: 1 > y')  -log1p(-y')
        #
        #  Part 2:
        #    -log(|1 + y'|)
        #  = (case 1: 1 + y' < 0)  -log(-y' - 1)
        #    (case 2: 1 + y' > 0)  -log1p(y')
        # (For cases where y' is numerically close to -1 or 1, clip gradient to 0 and don't apply log)
        y_prime = y * beta
        iladj = torch.zeros_like(y)
        # Part 1: case 1
        iladj[1. < y_prime] -= torch.log(y_prime[1. < y_prime] - 1.)
        # Part 1: case 2
        iladj[1. > y_prime] -= torch.log1p(-y_prime[1. > y_prime])
        # Part 2: case 1
        iladj[1. + y_prime < 0.] -= torch.log(-y_prime[1. + y_prime < 0.] - 1.)
        # Part 2: case 2
        iladj[1. + y_prime > 0.] -= torch.log1p(y_prime[1. + y_prime > 0.])

        _validate_tensor(iladj, "Outlier removal ILDJ: ")

        return torch.sum(iladj, axis=-1)

class AdaptivePowerTransform(dist.torch_transform.TransformModule):
    """
    Todo
    """

    # setup params
    domain = dist.transforms.constraints.real_vector # or just real?
    codomain = dist.transforms.constraints.real_vector
    bijective = True

    def __init__(self, input_dim, init_sigma=0.1, eps=1e-3):
        super(AdaptivePowerTransform, self).__init__(cache_size=1)

        self.input_dim = input_dim
        # learned standard deviation should be positive
        self.lambd = nn.Parameter(
            1. + torch.randn(self.input_dim) * init_sigma
        )
        self.eps = eps

    def _params(self):
        return self.lambd

    def _call(self, x):
        """
        Return input transformed by Yeo-Johnson transform using
        stored lambda parameter, independetly for each input dimension
        """

        _validate_tensor(self.lambd, "Power transform lambda: ")
        _validate_tensor(x, "Power transform input: ")

        out = torch.zeros_like(x)
        pos_x = x >= 0.                               # binary mask on input
        pos_l0 = torch.abs(self.lambd) < self.eps     # binary mask on lambda == 0
        pos_l2 = torch.abs(self.lambd - 2) < self.eps # binary mask on lambda == 2

        # Case 1: lambda != 0 and x >= 0
        if torch.any(pos_x & ~pos_l0):
            out[pos_x & ~pos_l0] = ((torch.pow(torch.abs(x) + 1., self.lambd) - 1) / self.lambd)[pos_x & ~pos_l0]
        # Case 2: lambda == 0 and x >= 0
        if torch.any(pos_x & pos_l0):
            out[pos_x & pos_l0] = torch.log1p(torch.abs(x[pos_x & pos_l0]))
        # Case 3: lambda != 2 and x < 0
        if torch.any(~pos_x & ~pos_l2):
            out[~pos_x & ~pos_l2] = ((torch.pow(1. + torch.abs(x), 2. - self.lambd) - 1.) / (self.lambd - 2.))[~pos_x & ~pos_l2]
        # Case 4: lambda == 2 and x < 0
        if torch.any(~pos_x & pos_l2):
            out[~pos_x & pos_l2] = -torch.log1p(torch.abs(x[~pos_x & pos_l2]))

        # print("Max and min")
        # print(torch.max(out))
        # print(torch.min(out))
        # print("Output:")
        # print(out)
        _validate_tensor(out, "Power Transform forward: ")

        return out

    def _inverse(self, y):
        # TODO: apply the same bugfixes to this code as was done for forward call
        raise NotImplementedError("Faulty implementation that may cause numerical errors.")
        out = torch.zeros_like(y)
        pos_y = y >= 0.                               # binary mask on input
        pos_l0 = torch.abs(self.lambd) < self.eps     # binary mask on lambda == 0
        pos_l2 = torch.abs(self.lambd - 2) < self.eps # binary mask on lambda == 2

        # Case 1: lambda != 0 and x >= 0
        out[pos_y & ~pos_l0] = (torch.pow(y * self.lambd + 1., 1 / self.lambd) - 1.)[pos_y & ~pos_l0]
        # Case 2: lambda == 0 and x >= 0
        out[pos_y & pos_l0] = torch.exp(y[pos_y & pos_l0]) - 1.
        # Case 3: lambda != 2 and x < 0
        out[~pos_y & ~pos_l2] = (1. - torch.pow(y * (self.lambd - 2.) + 1., 1 / (2. - self.lambd)))[~pos_y & ~pos_l2]
        # Case 4: lambda == 2 and x < 0
        out[~pos_y & pos_l2] = 1. - torch.exp(-y[~pos_y & pos_l2])

        return out

    def log_abs_det_jacobian(self, x, _):
        """
        Calculates the elementwise determinant of the log Jacobian

        :return: torch.tensor of same same as x
        """
        # TODO: apply the same bugfixes to this code as was done for inverse log abs det jacobian
        raise NotImplementedError("Faulty implementation that may cause numerical errors.")
        out = torch.zeros_like(x)
        pos_x = x >= 0.                               # binary mask on input
        pos_l0 = torch.abs(self.lambd) < self.eps     # binary mask on lambda == 0
        pos_l2 = torch.abs(self.lambd - 2) < self.eps # binary mask on lambda == 2

        # Case 1: lambda != 0 and x >= 0
        out[pos_x & ~pos_l0] = ((self.lambd - 1.) * torch.log1p(x))[pos_x & ~pos_l0]
        # Case 2: lambda == 0 and x >= 0
        out[pos_x & pos_l0] = -torch.log1p(x[pos_x & pos_l0])
        # Case 3: lambda != 2 and x < 0
        out[~pos_x & ~pos_l2] = ((1. - self.lambd) * torch.log1p(-x))[~pos_x & ~pos_l2]
        # Case 4: lambda == 2 and x < 0
        out[~pos_x & pos_l2] = -torch.log1p(-x[~pos_x & pos_l2])

        # apply determinant (which is sum of diagonals because logs)
        return torch.sum(out, axis=-1)

    def _inverse_log_abs_det_jacobian(self, _, y):
        out = torch.zeros_like(y)

        pos_x = y >= 0.                               # binary mask on input
        pos_l0 = torch.abs(self.lambd) < self.eps     # binary mask on lambda == 0
        pos_l2 = torch.abs(self.lambd - 2) < self.eps # binary mask on lambda == 2

        _validate_tensor(y, "Power transform ILDJ input: ")

        # Case 1: lambda != 0 and x >= 0
        pos_extra = torch.abs(y) * self.lambd < -1. + self.eps # to avoid NaNs in log1p
        if torch.any(pos_x & ~pos_l0 & ~pos_extra):
            out[pos_x & ~pos_l0 & ~pos_extra] = ((1. - self.lambd) / self.lambd * torch.log1p(torch.abs(y) * self.lambd))[pos_x & ~pos_l0 & ~pos_extra]
        if torch.any(pos_x & ~pos_l0 & pos_extra):
            out[pos_x & ~pos_l0 & pos_extra] = 0.

        # Case 2: lambda == 0 and x >= 0
        if torch.any(pos_x & pos_l0):
            out[pos_x & pos_l0] = y[pos_x & pos_l0]

        # Case 3: lambda != 2 and x < 0
        pos_extra = torch.abs(y) * (2. - self.lambd) < -1. + self.eps # to avoid NaNs in log1p
        if torch.any(~pos_x & ~pos_l2 & ~pos_extra):
            out[~pos_x & ~pos_l2 & ~pos_extra] = ((self.lambd - 1.) / (2. - self.lambd) * torch.log1p(torch.abs(y) * (2. - self.lambd)))[~pos_x & ~pos_l2 & ~pos_extra]
        if torch.any(~pos_x & ~pos_l2 & pos_extra):
            out[~pos_x & ~pos_l2 & pos_extra] = 0.

        # Case 4: lambda == 2 and x < 0
        if torch.any(~pos_x & pos_l2):
            out[~pos_x & pos_l2] = -y[~pos_x & pos_l2]

        _validate_tensor(out, "Power transform ILDJ: ")
        # apply determinant (which is sum of diagonals because logs
        return torch.sum(out, axis=-1)


class InvertBijector(dist.torch_transform.TransformModule):

     # setup params
    domain = dist.transforms.constraints.real_vector # or just real?
    codomain = dist.transforms.constraints.real_vector
    bijective = True

    def __init__(self, bijector):
        super(InvertBijector, self).__init__()
        self.bijector = bijector
        # check if has implemented ildj
        invert_op = getattr(self.bijector, "_inverse_log_abs_det_jacobian", None)
        if not callable(invert_op):
            raise ValueError("Provided bijector does not implement _inverse_log_abs_det_jacobian, so it cannot be inverted")

    def _params(self):
        return self.bijector._params()

    def _call(self, x):
        return self.bijector.inv(x)

    def _inverse(self, y):
        return self.bijector(y)

    def log_abs_det_jacobian(self, x, y):
        return self.bijector._inverse_log_abs_det_jacobian(y, x)


class AdaptivePreprocessingLayer(dist.torch_transform.TransformModule):
    domain = dist.transforms.constraints.real_vector # or just real?
    codomain = dist.transforms.constraints.real_vector
    bijective = True

    def __init__(self, input_dim, init_sigma=0.1, eps=1e-6, invert_bijector=True, adaptive_shift=True, adaptive_scale=True, adaptive_outlier_removal=True, adaptive_power_transform=True, outlier_removal_residual_connection=False, outlier_removal_mode='softplus'):
        super(AdaptivePreprocessingLayer, self).__init__(cache_size=1)

        self.input_dim = input_dim

        # initialise all the layers
        self.shift = AdaptiveShift(input_dim, init_sigma)
        self.scale = AdaptiveScale(input_dim, init_sigma)
        self.outlier_removal = AdaptiveOutlierRemoval(input_dim, init_sigma, outlier_removal_residual_connection, outlier_removal_mode)
        self.power_transfrom = AdaptivePowerTransform(input_dim, init_sigma, eps)

        # create the list of transformations
        transform_list = [
            self.outlier_removal,
            self.shift,
            self.scale,
            self.power_transform
        ]
        if invert_bijector:
            transform_list = [InvertBijector(t) for t in reversed(transform_list)]

        # compose our final bijector for internal use
        self.bijector = dist.torch_transform.ComposeTransformModule(transform_list)


    def _params(self):
        return self.shift._params(), self.scale._params(), self.outlier_removal._params(), self.power_transform._params()


    def _call(self, x):
        return self.bijector(x)


    def _inverse(self, x):
        return self.bijector.inv(x)


    def log_abs_det_jacobian(self, x, y):
        return self.bijector.log_abs_det_jacobian(x, y)


    def to(self, **kwargs):
        # to ensure bijector is moved to specified device when doing .to(DEV)
        module = super(AdaptivePreprocessingLayerm, self).to(**kwargs)
        module.bijector = self.bijector.to(**kwargs)
        return module


    def get_norm_flow(dev):
        base_dist = dist.Normal(torch.zeros(self.input_dim, device=dev), torch.ones(self.input_dim, device=dev)).to_event(1)
        return dist.TransformedDistribution(base_dist, [self])

    def get_optimizer_param_dict(base_lr, scale_lr, shift_lr, outlier_lr, power_lr):
        return {
            'parameters' : self.
        }
