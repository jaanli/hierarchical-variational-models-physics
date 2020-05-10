import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from network import mask
from network import  masked_linear as net


class MADE(nn.Module):
  """Implements MADE: Masked Autoencoder for Distribution Estimation.

  Follows http://proceedings.mlr.press/v37/germain15.pdf
  """
  def __init__(self, 
               num_input, 
               num_output, 
               num_hidden,
               use_context,
               hidden_degrees,
               activation):
    super().__init__()
    assert num_output % num_input == 0, 'Outputs must be multiple of inputs'
    self.use_context = use_context
    masks = mask.create_masks(mask.create_degrees(
        input_size=num_input,
        hidden_units=[num_hidden] * 2,
        input_order='left-to-right',
        hidden_degrees=hidden_degrees))
    masks[-1] = np.hstack([masks[-1] for _ in range(num_output // num_input)])
    masks = [torch.from_numpy(mask.astype(np.float32).T) for mask in masks]
    if num_input <= 64:
      mask.check_masks(masks)
    if use_context:
      if activation == 'relu':
        act_fn = lambda: ContextReLU(inplace=True)
      elif activation == 'prelu':
        act_fn = lambda: ContextPReLU(num_hidden, init=0.5)
      modules = [net.ConditionalMaskedLinear(num_input, num_hidden, masks[0], num_input),
                 act_fn(),
                 net.ConditionalMaskedLinear(num_hidden, num_hidden, masks[1], num_input),
                 act_fn(),
                 net.ConditionalMaskedLinear(num_hidden, num_output, masks[2], num_input)]
      self.net = ContextSequential(*modules)
    else:
      if activation == 'relu':
        act_fn = lambda: nn.ReLU(inplace=True)
      elif activation == 'prelu':
        act_fn = lambda: nn.PReLU(num_hidden, init=0.5)
      modules = [net.MaskedLinear(num_input, num_hidden, masks[0]),
                 act_fn(),
                 net.MaskedLinear(num_hidden, num_hidden, masks[1]),
                 act_fn(),
                 net.MaskedLinear(num_hidden, num_output, masks[2])]
      self.net = nn.Sequential(*modules)
    lower_diag = torch.tril(torch.ones(num_input, num_input), diagonal=-1)
    lower_diag = lower_diag.repeat(num_output // num_input, 1)
    self.residual_linear = net.MaskedLinear(num_input, num_output, lower_diag, bias=False)

  def forward(self, input, context=None):
    if self.use_context:
      return self.net(input, context) + self.residual_linear(input)
    else:
      return self.net(input) + self.residual_linear(input)


class BasicBlock(nn.Module):
  def __init__(self, num_input, mask):
    self.relu = nn.ReLU(inplace=True)
    self.linear = MaskedLinear(num_input, num_input, mask)


class ContextReLU(nn.Module):
  def __init__(self, inplace):
    super().__init__()
    self.relu = nn.ReLU(inplace=inplace)

  def forward(self, input, context):
    return self.relu(input)


class ContextSoftplus(nn.Module):
  def __init__(self):
    super().__init__()
    self.softplus = nn.Softplus()

  def forward(self, input, context):
    return self.softplus(input)


class ContextPReLU(nn.Module):
  def __init__(self, num_parameters, init):
    super().__init__()
    self.prelu = nn.PReLU(num_parameters=num_parameters, init=init)

  def forward(self, input, context):
    return self.prelu(input)


class ContextSequential(nn.Sequential):
  """Allow nn.Sequential to take multiple inputs."""
  
  def forward(self, input, context=None):
    for block in self._modules.values():
      input = block(input, context)
    return input
