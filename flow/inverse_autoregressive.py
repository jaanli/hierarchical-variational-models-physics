import math
import torch
from torch import nn

import network


class AutoregressiveSampleAndLogProb(nn.Module):
  """Inverse Autoregressive Flows with LSTM-type update.
  
  Eq 11-14 of https://arxiv.org/abs/1606.04934
  """
  def __init__(self,
               num_input,
               use_context,
               hidden_size,
               hidden_degrees,
               activation,
               flow_std):
    super().__init__()
    self.made = network.MADE(num_input=num_input, 
                             num_output=num_input * 2,
                             num_hidden=hidden_size, 
                             use_context=use_context, 
                             hidden_degrees=hidden_degrees,
                             activation=activation)
    # init such that sigmoid(s) is close to 1 for stability
    self.sigmoid_arg_bias = nn.Parameter(torch.ones(num_input) * 2)
    self.sigmoid = nn.Sigmoid()
    self.log_sigmoid = nn.LogSigmoid()
    self.scale = 1.0
    self.flow_std = flow_std

  def forward(self, input, context=None):
    """Returns:
      - Transformed input variable z = f(input)
      - log density of the transformed variable corresponding to f
    """
    out = self.made(input, context)
    m, s = torch.chunk(self.made(input, context), chunks=2, dim=-1)
    s = s + self.sigmoid_arg_bias
    sigmoid = self.sigmoid(s)
    z = sigmoid * input + (1 - sigmoid) * m
    # inverse is g(z) = z / scale
    # => correction to probability is g'(z) = 1/scale
    return z * self.scale, (-self.log_sigmoid(s) - math.log(self.scale)).sum(-1)

  @torch.no_grad()
  def initialize_scale(self, input, context=None):
    z, log_det = self.forward(input, context)
    self.scale = self.flow_std / z.std() 
    print('IAF output std: %.3f' % z.std())
    print('Multiplying output of flow by: %.3f' % self.scale)
    return z, log_det



class InverseAutoregressive(nn.Module):
  """Inverse Autoregressive Flows with regular update.
  
  Eq 11-14 of https://arxiv.org/abs/1606.04934
  """
  def __init__(self,
               num_input,
               use_context,
               hidden_size,
               hidden_degrees,
               activation,
               flow_std):
    super().__init__()
    self.made = network.MADE(num_input=num_input, 
                             num_output=num_input * 2,
                             num_hidden=hidden_size, 
                             use_context=use_context, 
                             hidden_degrees=hidden_degrees,
                             activation=activation)
    self.scale = 1.0
    self.flow_std = flow_std

  def forward(self, input, context=None):
    out = self.made(input, context)
    m, s = torch.chunk(self.made(input, context), chunks=2, dim=-1)
    z = m + torch.exp(-s) * input
    # inverse is g(z) = z / scale
    # => correction to probability is g'(z) = 1/scale
    return z * self.scale, s - math.log(self.scale)

  @torch.no_grad()
  def initialize_scale(self, input, context=None):
    z, log_det = self.forward(input, context)
    self.scale = self.flow_std / z.std() 
    print('IAF output std: %.3f' % z.std())
    print('Multiplying output of flow by: %.3f' % self.scale)
    return z, log_det
