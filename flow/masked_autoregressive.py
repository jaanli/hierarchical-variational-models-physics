import math
import torch
from torch import nn

import network
      
class AutoregressiveInverseAndLogProb(nn.Module):
  """Use MADE to build MAF: Masked Autoregressive Flow.

  Implements Eqs 2-5 in https://arxiv.org/abs/1705.07057
  """
  def __init__(self,
               num_input,
               use_context,
               use_tanh,
               hidden_size,
               hidden_degrees,
               flow_std,
               activation):
    super().__init__()
    self.f_mu_alpha = network.MADE(num_input=num_input, 
                                   num_output=num_input * 2, 
                                   use_context=use_context,
                                   num_hidden=hidden_size,
                                   hidden_degrees=hidden_degrees,
                                   activation=activation)
    self.use_tanh = use_tanh
    self.scale = 1.0
    self.flow_std = flow_std

  @torch.no_grad()
  def initialize_scale(self, input, context=None):
    u, log_det = self.forward(input, context)
    self.scale = self.flow_std / u.std()
    print('MAF output std: %.3f' % u.std())
    print('Multiplying output of flow by: %.3f' % self.scale)
    return u, log_det

  def forward(self, input, context=None):
    """Returns:
      - random numbers u used to generate an input: input = f(u)
      - log density of input corresponding to transform f

    Prob correction is log det |\partial_x f^{-1}|."""
    # Calculate u = f^{-1}(x) with equations 4-5
    # MAF parameterizes the forward direction using the inverse
    x = input
    mu, alpha = torch.chunk(self.f_mu_alpha(x, context), chunks=2, dim=-1)
    if self.use_tanh:
      alpha = torch.tanh(alpha)
    u = (x - mu) * torch.exp(-alpha)
    # u is the output of the inverse, so rescaling adds |d / du (scale * u)|
    # to the density
    return u * self.scale, (-alpha + math.log(self.scale)).sum(-1)
