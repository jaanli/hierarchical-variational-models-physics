import math
import torch
from torch import nn
from torch.nn import functional as F


class MaskedLinear(nn.Module):
  """Linear layer with some input-output connections masked."""
  def __init__(self, in_features, out_features, mask, bias=True):
    super().__init__()
    self.linear = nn.Linear(in_features, out_features, bias)
    self.register_buffer("mask", mask)
    self.reset_parameters()

  def reset_parameters(self):
    if self.linear.bias is not None:
      nn.init.constant_(self.linear.bias, 0)
    gain = nn.init.calculate_gain('relu')
    # n_l is number of inputs in kaiming init: https://arxiv.org/pdf/1502.01852v1.pdf
    # weight matrix is of shape (out_features, in_features)
    # mask yields the number of inputs every output feature has
    # this gives correct kaiming initialization
    n_l = self.mask.sum(axis=1)
    std = gain / n_l.sqrt()
    inf_mask = torch.isinf(std)
    std[inf_mask] = 1
    out_size, in_size = self.linear.weight.shape
    std = std.view(out_size, 1)
    with torch.no_grad():
      self.linear.weight.normal_(0, 1)
      self.linear.weight.mul_(std)
  
  def forward(self, input):
    return F.linear(input, self.mask * self.linear.weight, self.linear.bias)


class ResidualMaskedLinear(MaskedLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, input):
    return super().forward(input) + input @ self.mask.t()


class ConditionalMaskedLinear(nn.Module):
  """Use a different mask for conditioning set."""
  def __init__(self, in_features, out_features, mask, context_features):
    super().__init__()
    self.linear = MaskedLinear(in_features, out_features, mask)
    self.cond_linear = nn.Linear(context_features, out_features, bias=False)

  def forward(self, input, context):
    return self.linear(input) + self.cond_linear(context)


class MaskedConditionalMaskedLinear(nn.Module):
  """Use a different mask for conditioning set."""
  def __init__(self, in_features, out_features, mask, conditional_mask):
    super().__init__()
    self.linear = MaskedLinear(in_features, out_features, mask)
    self.conditional_linear = MaskedLinear(in_features, out_features, conditional_mask, bias=False)

  def forward(self, input, context):
    return self.linear(input) + self.conditional_linear(context)


class ResidualConditionalMaskedLinear(ConditionalMaskedLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, input, context):
    return super().forward(input, context) + input @ self.linear.mask.t()
