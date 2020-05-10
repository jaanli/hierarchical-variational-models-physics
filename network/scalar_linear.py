""""Neural network modules. 

Largely from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import math
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class ElementwiseLinear(nn.Module):
  """Linear transformation applied to every dimension of the input independently."""
  def __init__(self, in_features, out_features):
    super().__init__()
    if out_features == 1:
      shape = (in_features,)
    else:
      shape = (in_features, out_features)
    self.weight = nn.Parameter(torch.Tensor(*shape))
    self.bias = nn.Parameter(torch.Tensor(*shape))

  @torch.no_grad()
  def reset_parameters(self):
    gain = init.calculate_gain(nonlinearity='relu')
    # use fan-out as in pytorch resnet initialization
    fan_out = self.weight.size(0)
    std = gain / math.sqrt(fan_out)
    print(f'Initializing ElementwiseLinear weights to Normal(0, {std:.3f})')
    self.weight.normal_(0, std)
    bound = 1 / math.sqrt(fan_out)
    print(f'Initializing ElementwiseLinear bias to uniform(-{bound:.3f}, {bound:.3f})')
    init.uniform_(self.bias, -bound, bound)

  def forward(self, input):
    # input: (num_eps_samples, latent_size)
    return input * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
