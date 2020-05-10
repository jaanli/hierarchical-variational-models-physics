import numpy as np
import torch
from torch import nn


class BatchNormalization(nn.Module):
  """Normalize a batch to have zero mean and unit variance when running in inverse mode.

  Why inverse? Masked Autoregressive Flows have cheap density
  evaluations because they parameterize the inverse.
 
  As described in Equations 21-23 of https://arxiv.org/abs/1705.07057

  """
  def __init__(self, num_features, momentum=0.0, eps=1e-5):
    super().__init__()
    self.eps = eps
    self.momentum = momentum
    self.beta = nn.Parameter(torch.zeros(num_features))
    self.log_gamma = nn.Parameter(torch.zeros(num_features))
    self.register_buffer("running_mean", torch.zeros(num_features))
    self.register_buffer("running_var", torch.ones(num_features))

  def reset_parameters(self):
    # log_gamma is like a weight parameter
    nn.init.zeros_(self.log_gamma)
    # beta is like a bias
    nn.init.zeros_(self.beta)
    self.running_mean.zero_()
    self.running_var.fill_(1)

  def forward(self, input, context=None):
    x = input
    if self.training:
      self.batch_mean = torch.mean(x, dim=0)
      self.batch_var = torch.var(x, dim=0)
      self.running_mean.mul_(self.momentum)
      self.running_var.mul_(self.momentum)
      self.running_mean.add_((1. - self.momentum) * self.batch_mean.data)
      self.running_var.add_((1. - self.momentum) * self.batch_var.data)
      m = self.batch_mean
      v = self.batch_var
    else:
      m = self.running_mean
      v = self.running_var
    u = (x - m) / torch.sqrt(v + self.eps) * torch.exp(self.log_gamma) + self.beta
    log_det_inv = self.log_gamma - 0.5 * torch.log(v + self.eps)
    return u, log_det_inv


class Reverse(nn.Module):
  """ An implementation of a reversing layer from
  Density estimation using Real NVP
  (https://arxiv.org/abs/1605.08803).

  From https://github.com/ikostrikov/pytorch-flows/blob/master/main.py
  """

  def __init__(self, num_input):
    super(Reverse, self).__init__()
    self.perm = np.array(np.arange(0, num_input)[::-1])
    self.inv_perm = np.argsort(self.perm)

  def forward(self, inputs, context=None, mode='forward'):
    return inputs[..., self.perm], 0


def inv_softplus(x):
  return np.log(np.exp(x) - 1)

