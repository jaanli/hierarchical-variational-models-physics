"""Recursive variational approximations.

Every posterior returns three things:
nu_0: (num_samples, latent_size), the random variable corresponding to the base density
log_r_nu_0: (num_samples, latent_size), the density of r(nu_0 | z)
log_r_nu: (num_samples,), the sum of the log density of r(nu_i) for i > 0"""
import torch
import torch.nn as nn
from torch.nn import init
import math
from torch import distributions as torch_dist

import flow
import network
import distributions
from util import reshape_lattice

class BinaryConditionalPosterior(nn.Module):
  def __init__(self,
               latent_size,
               flow_depth,
               flow_std,
               hidden_size,
               hidden_degrees,
               reverse,
               activation):
    super().__init__()
    self.log_r_nu_0 = BinaryDistribution(latent_size, scale=flow_std)
    modules = []
    for _ in range(flow_depth):
      modules.append(flow.AutoregressiveInverseAndLogProb(num_input=latent_size,
                                                   use_context=True,
                                                   use_tanh=True,
                                                   hidden_size=hidden_size,
                                                   hidden_degrees=hidden_degrees,
                                                   flow_std=flow_std,
                                                   activation=activation))
      if reverse:
        modules.append(flow.Reverse(latent_size))
    self.r_nu = flow.FlowSequential(*modules)
  
  def initialize_scale(self, nu_K, z):
    nu_0 = self.r_nu.initialize_scale(nu_K, context=z)
    with torch.no_grad():
      self.log_r_nu_0.log_scale.normal_(math.log(nu_0.std()), 0.01)
    print('r(nu_0) log std initialized with mean of: %.3f' % math.log(nu_0.std()))

  def forward(self, nu_K, z):
    nu_0, log_r_nu = self.r_nu(nu_K, context=z)
    log_r_nu_0 = self.log_r_nu_0(nu_0, z)
    return nu_0, log_r_nu_0, log_r_nu


class BinaryPosterior(nn.Module):
  def __init__(self,
               latent_size,
               flow_depth,
               flow_std,
               hidden_size,
               hidden_degrees,
               reverse,
               activation):
    super().__init__()
    self.log_r_nu_0 = BinaryDistribution(latent_shape=(latent_size,),
                                         scale=flow_std)
    modules = []
    for _ in range(flow_depth):
      modules.append(flow.AutoregressiveInverseAndLogProb(num_input=latent_size,
                                                   use_context=False,
                                                   use_tanh=True,
                                                   hidden_size=hidden_size,
                                                   hidden_degrees=hidden_degrees,
                                                   flow_std=flow_std,
                                                   activation=activation))
      if reverse:
        modules.append(flow.Reverse(latent_size))
    self.r_nu = flow.FlowSequential(*modules)
  
  def initialize_scale(self, nu_K, z):
    nu_0 = self.r_nu.initialize_scale(nu_K)
    with torch.no_grad():
      self.log_r_nu_0.log_scale.normal_(math.log(nu_0.std()), 0.01)
    print('r(nu_0) log std initialized with mean of: %.3f' % math.log(nu_0.std()))

  def inverse_and_log_prob(self, nu_K, z):
    nu_K = nu_K.reshape(nu_K.shape[0], -1)
    nu_0, log_r_nu = self.r_nu(nu_K)
    log_r_nu_0 = self.log_r_nu_0(nu_0, z)
    log_r_nu_0 = reshape_lattice(log_r_nu_0)
    return nu_0, log_r_nu_0, log_r_nu


class ConditionalPosterior(nn.Module):
  def __init__(self, 
               latent_size,
               flow_depth,
               flow_std,
               hidden_size,
               hidden_degrees,
               activation):
    super().__init__()
    self.r_nu_0 = torch_dist.Normal(loc=0, scale=1)
    self.register_buffer('zero', torch.Tensor([0]))
    modules = []
    for _ in range(flow_depth):
      modules.append(flow.AutoregressiveInverseAndLogProb(num_input=latent_size,
                                                   use_context=True,
                                                   use_tanh=True,
                                                   hidden_size=hidden_size,
                                                   hidden_degrees=hidden_degrees,
                                                   flow_std=flow_std,
                                                   activation=activation))
      modules.append(flow.Reverse(latent_size))
    self.r_nu = flow.FlowSequential(*modules)

  def initialize_scale(self, nu_K, z):
    nu_0 = self.r_nu.initialize_scale(nu_K, context=z)
    self.r_nu_0.scale = nu_0.std()
    print('r(nu_0) std set to: %.3f' % nu_0.std())

  def forward(self, nu_K, z):
    nu_0, log_r_nu = self.r_nu(nu_K, context=z)
    log_r_nu_0 = self.r_nu_0.log_prob(nu_0)
    return nu_0, self.zero, log_r_nu + log_r_nu_0



class MixturePosterior(nn.Module):
  """r(\nu | z) uses a mixture to approximate the terms that mix z and nu.

     First transform nu independently, then the last transformation is
     jointly using the mixture.
  """
  def __init__(self, latent_size, hidden_size, flow_depth, num_components):
    super().__init__()
    modules = []
    for _ in range(flow_depth):
      modules.append(flow.AutoregressiveInverseAndLogProb(
        num_input=latent_size,
        num_hidden=hidden_size,
        num_context=latent_size,
        use_tanh=True))
      modules.append(flow.Reverse(latent_size))
    self.r_nu_first = flow.FlowSequential(*modules)
    self.r_nu_last = flow.BinaryMixtureTransform(latent_size, num_components)
    self.log_r_0 = distributions.StandardNormalLogProb()

  def forward(self, nu_L_plus_K, z):
    # invert the joint transformation of nu and z, dimension-wise
    mask = (z > 0).float().unsqueeze(-1)
    nu_K, log_r_nu_K = self.r_nu_last(nu_L_plus_K, z)
    # invert the first K transformations of the flow
    nu_0, log_r_nu = self.r_nu_first(nu_K, context=z)
    log_r_0 = self.log_r_0(nu_0)
    return nu_0, log_r_nu_K + log_r_nu + log_r_0


class BinaryDistribution(nn.Module):
  """r(nu_0 | z) as a Gaussian for when z is binary."""
  def __init__(self, latent_shape, scale):
    super().__init__()
# TODO    raise ValueError('potentially wrong - how to account for binary log probability of thresholding?')
    self.loc = nn.Parameter(torch.Tensor(*(latent_shape + (2,))))
    self.log_scale = nn.Parameter(torch.Tensor(*(latent_shape + (2,))))
    self.scale = scale
    self._log_prob = distributions.NormalLogProb()
    self.shape = latent_shape
    self.reset_parameters()

  def reset_parameters(self):
    gain = init.calculate_gain(nonlinearity='relu')
    # use fan-out as in pytorch resnet initialization
    fan_out = self.loc.size(0) 
    std = gain / math.sqrt(fan_out)
    with torch.no_grad():
      print(f'Initializing r locations to Normal(0, {std:.3f})')
      self.loc.normal_(0, std)
      loc = math.log(self.scale)
      print(f'Initializing r scale parameters to Normal({loc:.3f}, {std:.3f})')
      self.log_scale.normal_(loc, std)

  def forward(self, nu, z):
    """Return inverse and log prob."""
    mask = (z > 0).float().unsqueeze(-1)
    mask = mask.view((-1,) + self.shape + (1,))
    # only one dimension will be active
    loc = (self.loc * mask).sum(-1)
    log_scale = (self.log_scale * mask).sum(-1)
    scale = torch.exp(log_scale)
    return self._log_prob(loc, scale, nu)


class ResidualFirstVariationalPosterior(nn.Module):
  """Parameterize r(nu_0 | z) as a Gaussian with mean and variance from a neural nets.

  Every neural net takes scalar input z_i (a single latent variable),
  and outputs a mean and variance for the Gaussian for nu_{0, i} (a single hierarchical latent).

  The neural nets are single-neuron resnets:
  https://papers.nips.cc/paper/7855-resnet-with-one-neuron-hidden-layers-is-a-universal-approximator.pdf
  """
  def __init__(self, latent_size):
    super().__init__()
    self.net = network.ScalarSingleNeuronResNet(block=network.ScalarSingleNeuronBasicBlock, 
                                                layer_num_blocks=[1, 1, 1, 1], 
                                                input_size=latent_size, 
                                                outputs_per_input=2)
    self._softplus = nn.Softplus()
    self._log_prob = distributions.NormalLogProb()

  def log_prob(self, nu_0, z):
    loc, inv_sp_scale = self.net(z).chunk(chunks=2, dim=-1)
    scale = self._softplus(inv_sp_scale.squeeze())
    return self._log_prob(loc.squeeze(), scale, nu_0)


class RealNVPPosterior(nn.Module):
  """r(\nu | z) is a RealNVP flow.
  """
  def __init__(self,
               latent_shape,
               flow_depth,
               flow_std,
               hidden_size):
    super().__init__()
    modules = [flow.CheckerSplit(latent_shape)]
    for flow_num in range(flow_depth):
      modules.append(flow.RealNVPPermuteInverseAndLogProb(
        in_channels=1,
        hidden_size=hidden_size,
        # invert mask opposite to prior 
        parity=True if flow_num % 2 == 0 else False))
    modules.append(flow.CheckerConcat(latent_shape))      
    self.r_nu = flow.RealNVPSequential(*modules)
    self.log_r_nu_0 = BinaryDistribution(latent_shape=latent_shape,
                                         scale=flow_std)

  def inverse_and_log_prob(self, nu_K, z):
    nu_0, log_r_nu = self.r_nu(nu_K)
    log_r_nu_0 = self.log_r_nu_0(nu_0, z)
    return nu_0, log_r_nu_0, log_r_nu


class RealNVPFullConditioningPosterior(nn.Module):
  """r(\nu | z) is a RealNVP flow."""
  def __init__(self,
               latent_shape,
               flow_depth,
               flow_std,
               hidden_size):
    super().__init__()
    self.split = flow.CheckerSplit(latent_shape)
    self.concat = flow.CheckerConcat(latent_shape)
    modules = []
    for flow_num in range(flow_depth):
      modules.append(flow.RealNVPPermuteInverseAndLogProb(
        # input: nu, z_transf (black squares of checkerboard) and z_const (white squares)
        in_channels=3,
        hidden_size=hidden_size,
        # invert mask opposite to prior 
        parity=True if flow_num % 2 == 0 else False))
    self.r_nu = flow.SplitSequential(*modules)
    self.log_r_0 = distributions.StandardNormalLogProb()

  def inverse_and_log_prob(self, nu_K, z):
    nu_K_transf, nu_K_const = self.split(nu_K)
    z_transf, z_const = self.split(z)
    nu_0_transf, nu_0_const, log_r_nu = self.r_nu(nu_K_transf, 
                                                  nu_K_const, 
                                                  context=[z_transf, z_const])
    nu_0 = self.concat(nu_0_transf, nu_0_const)
    log_r_nu_0 = self.log_r_0(nu_0).sum((1, 2))
    return nu_0, None, log_r_nu + log_r_nu_0
