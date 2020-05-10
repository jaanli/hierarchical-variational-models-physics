import math
import torch
import numpy as np
from torch import nn

import network



class RealNVPInverseAndLogProb(nn.Module):
  """Checkerboard masking with convolution for nearest neighbors.

  Implements Eq. 9 in https://arxiv.org/pdf/1605.08803.pdf
  """
  def __init__(self, hidden_size, kernel_size, mask, condition_on_latents=False):
    super().__init__()
    # use kernel_size = (3, 3) for nearest neighbors
    self.s = network.Conv2d(hidden_sizes=[hidden_size],
                            use_final_bias=False,
                            inner_activation='leaky_relu',
                            final_activation='tanh',
                            use_dropout=0,
                            use_weight_norm=False,
                            kernel_size=kernel_size)
    self.t = network.Conv2d(hidden_sizes=[hidden_size],
                            use_final_bias=False,
                            inner_activation='leaky_relu',
                            final_activation=None,
                            use_dropout=0,
                            use_weight_norm=False,
                            kernel_size=kernel_size)    
    self.register_buffer('mask', mask)
    if condition_on_latents:
      self.conditioning_net = network.Conv2d(
        hidden_sizes=[hidden_size],
        use_final_bias=False,
        inner_activation='leaky_relu',
        final_activation=None,
        use_dropout=0,
        use_weight_norm=False,
        kernel_size=kernel_size)    
      

  def forward(self, x, context=None):
    """Returns:
      - random numbers y used to generate x, y = f(x)
      - log determinant Jacobian of f; log probability of inverse x
    """
    b = self.mask
    s = self.s(b * x)
    # calculate random numbers y using f^{-1}
    inverse = (b * x + (1 - b) * (x * torch.exp(s) + self.t(b * x)))
    # log density of x correction from f is log det |df/dx|
    return inverse, (s * (1 - b)).sum((1, 2))

  def inverse(self, y, context=None):
    """Returns:
      - x generated with random numbers y, x = f^{-1}(y)
      - log determinant Jacobian of f, log probability of y
    """
    b = self.mask
    s = self.s(b * y)
    sample = (b * y) + (1 - b) * (y - self.t(b * y)) * torch.exp(-s)
    return sample, (s * (1 - b)).sum((1, 2))


class RealNVPDilatedInverseAndLogProb(nn.Module):
  """Use checkerboard masking by manipulating convolution strides and rolling the input.
  """
  def __init__(self, hidden_size):
    super().__init__()
    self.s = network.Conv2dDilatedCheckerboard(hidden_size=16,
                                        inner_activation='leaky_relu',
                                        final_activation='tanh')
    self.t = network.Conv2dDilatedCheckerboard(hidden_size=16,
                                        inner_activation='leaky_relu',
                                        final_activation=None)

  def forward(self, x, context=None):
    """See figure 3 of https://arxiv.org/pdf/1605.08803.pdf

    This transforms the white squares using the output of the networks
    applied to the black squares.
    """
    x_roll = torch.empty(x.shape).to(x.device)
    # even rows stay the same
    x_roll[:, ::2, :] = x[:, ::2, :]
    # roll the odd rows to the left by 1
    x_roll[:, 1::2, :] = x[:, 1::2, :].roll(-1, dims=-1)
    s = self.s(x_roll)
    # every second column is corresponds to the checkerboard
    inverse = x_roll.clone()
    inverse[..., 1::2] = x_roll[..., 1::2] * torch.exp(s) + self.t(x_roll)
    return inverse, s.sum((1, 2))
    
  def inverse(self, y, context=None):
    s = self.s(y)
    sample = y
    sample[..., 1::2] = (y[..., 1::2] - self.t(y)) * torch.exp(-s)
    sample[:, 1::2, :] = sample[:, 1::2, :].roll(1, dims=-1)    
    return sample, s.sum((1, 2))


class RealNVPFastInverseAndLogProb(nn.Module):
  """Use checkerboard masking by manipulating convolution strides and rolling the input.
  """
  def __init__(self, hidden_size, parity=False):
    super().__init__()
    self.s = network.Conv2dRect(in_channels=1, 
                                hidden_size=hidden_size,
                                inner_activation='leaky_relu',
                                final_activation='tanh')
    self.t = network.Conv2dRect(in_channels=1,
                                hidden_size=hidden_size,
                                inner_activation='leaky_relu',
                                final_activation=None)
    self.parity = parity

  def forward(self, x, context=None):
    """See figure 3 of https://arxiv.org/pdf/1605.08803.pdf

    This transforms the white squares using the output of the networks
    applied to the black squares.
    """
    x_roll = torch.empty(x.shape).to(x.device)
    # even rows stay the same
    x_roll[:, ::2, :] = x[:, ::2, :]
    # roll the odd rows to the left by 1
    x_roll[:, 1::2, :] = x[:, 1::2, :].roll(-1, dims=-1)    
    transformed, const = x_roll[..., 1::2], x_roll[..., ::2]
    if self.parity:
      transformed, const = const, transformed
    s = self.s(const)
    inverse = torch.empty(x_roll.shape).to(x_roll.device)
    # even columns stay the same
    inverse[..., ::2] = const
    # odd columns are transformed
    inverse[..., 1::2] = transformed * torch.exp(s) + self.t(const)
    # roll odd rows to right by 1 (undo the previous roll)
    inverse[:, 1::2, :] = inverse[:, 1::2, :].roll(1, dims=-1)
    return inverse, s.sum((1, 2))
    
  def inverse(self, y, context=None):
    # undo the last roll
    y[:, 1::2, :] = y[:, 1::2, :].roll(-1, dims=-1)
    transformed, const = y[..., 1::2], y[..., ::2]
    s = self.s(const)
    sample = torch.empty(y.shape).to(y.device)
    # even columns stay the same 
    sample[..., ::2] = const
    # odd columns are transformed
    sample[..., 1::2] = (transformed - self.t(const)) * torch.exp(-s)
    # undo parity
    if self.parity:
      # order of assignment is important, replacing const by sample[..., ::2] breaks
      sample[..., ::2], sample[..., 1::2] = sample[..., 1::2], const
    # undo the first roll
    sample[:, 1::2, :] = sample[:, 1::2, :].roll(1, dims=-1)
    return sample, s.sum((1, 2))


class CheckerboardSplit(nn.Module):
  def forward(self, input):
    input[..., 1::2, :] = input[..., 1::2, :].roll(-1, dims=-1)
    return input[..., ::2], input[..., 1::2]

  def inverse(self, even, odd):
    res = torch.cat([even, odd], dim=-1)
    L = res.shape[-1]
    perm = np.argsort(list(range(L)[::2]) + list(range(L)[1::2]))
    out = res[..., perm]
    out[..., 1::2, :] = out[..., 1::2, :].roll(1, dims=-1)
    return out

class CheckerboardConcat(CheckerboardSplit):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.forward, self.inverse = self.inverse, self.forward


class CheckerSplit(nn.Module):
  """Split so first and second half are white/black checkerboard parts."""
  def __init__(self, latent_shape):
    super().__init__()
    self.latent_shape = latent_shape
    self.half_lattice = (latent_shape[0], latent_shape[1] // 2)
    idx = np.arange(np.prod(latent_shape)).reshape(latent_shape)
    checker_idx = idx.copy()
    checker_idx[1::2] = np.roll(idx[1::2], -1, axis=-1)
    even_idx = checker_idx[..., ::2].ravel()
    odd_idx = checker_idx[..., 1::2].ravel()
    # permuted indices with even cols are first, odd cols last
    self.permute_idx = np.concatenate((even_idx, odd_idx))
    self.unpermute_idx = np.argsort(self.permute_idx)

  def forward(self, x, context=None):
    """Input: (num_samples, L, L)
      Output: (num_samples, L ** 2), 0 (Jacobian is zero for this transform)
    First/second half of output is white/black squares on checkerboard.
    """
    num_samples = x.shape[0]
    even, odd = x.view(num_samples, -1)[..., self.permute_idx].chunk(2, dim=-1)
    shape = (num_samples,) + self.half_lattice
    return even.view(shape), odd.view(shape)


class CheckerConcat(CheckerSplit):

  def forward(self, transf, const, context=None):
    transf = transf.view(transf.shape[0], -1)
    const = const.view(const.shape[0], -1)
    res = torch.cat((transf, const), dim=-1)
    return res[..., self.unpermute_idx].view(res.shape[0], *self.latent_shape)


class RealNVPPermuteInverseAndLogProb(nn.Module):
  def __init__(self, in_channels, hidden_size, parity=False):
    super().__init__()
    self.s = network.Conv2dRect(in_channels=in_channels,
                                hidden_size=hidden_size,
                                inner_activation='leaky_relu',
                                final_activation='tanh')
    self.t = network.Conv2dRect(in_channels=in_channels,
                                hidden_size=hidden_size,
                                inner_activation='leaky_relu',
                                final_activation=None)
    self.parity = parity

  def forward(self, transf, const, context=None):
    if self.parity:
      transf, const = const, transf
    s = self.s(const, context)
    return transf * torch.exp(s) + self.t(const, context), const, s.sum((1, 2))

  def inverse(self, transf, const, context=None):
    s = self.s(const)
    res = (transf - self.t(const, context)) * torch.exp(-s)
    if self.parity:
      return const, res, s.sum((1, 2))
    else:
      return res, const, s.sum((1, 2))


class RealNVPPermuteSampleAndLogProb(RealNVPPermuteInverseAndLogProb):
  def forward(self, transf, const, context=None):
    """Returns:
      - x generated with random numbers y, with x = f(y)
      - log det Jacobian of f^{-1} (the negation); log prob of sample x
    """
    transf, const, log_det_jacobian_f = super().forward(transf, const, context)
    return transf, const, -log_det_jacobian_f

  def inverse(self, transf, const, context=None):
    """Returns:
      - random numbers y used to generate input x, y = f^{-1}(x)
      - log det Jacobian of f^{-1}; log prob of sample x
    """
    transf, const, log_det_jacobian_f = super().inverse(transf, const, context)
    return transf, const, -log_det_jacobian_f



class RealNVPSampleAndLogProb(RealNVPInverseAndLogProb):
  def forward(self, y, context=None):
    """Returns:
      - x generated with random numbers y, with x = f(y)
      - log det Jacobian of f^{-1} (the negation); log prob of sample x
    """
    sample, log_det_jacobian_f = super().forward(y, context)
    return sample, -log_det_jacobian_f

  def inverse(self, x, context=None):
    """Returns:
      - random numbers y used to generate input x, y = f^{-1}(x)
      - log det Jacobian of f^{-1}; log prob of sample x
    """
    inverse, log_det_jacobian_f = super().inverse(x, context)
    return inverse, -log_det_jacobian_f


class RealNVPFastSampleAndLogProb(RealNVPFastInverseAndLogProb):
  def forward(self, y, context=None):
    """Returns:
      - x generated with random numbers y, with x = f(y)
      - log det Jacobian of f^{-1} (the negation); log prob of sample x
    """
    sample, log_det_jacobian_f = super().forward(y, context)
    return sample, -log_det_jacobian_f

  def inverse(self, x, context=None):
    """Returns:
      - random numbers y used to generate input x, y = f^{-1}(x)
      - log det Jacobian of f^{-1}; log prob of sample x
    """
    inverse, log_det_jacobian_f = super().inverse(x, context)
    return inverse, -log_det_jacobian_f
