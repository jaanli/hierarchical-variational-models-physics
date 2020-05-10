"""Variational priors q(nu)."""
import math
import torch
from torch import nn
from torch import distributions
import numpy as np

import flow
import network.mask
from util import reshape_lattice


class AutoregressivePrior(nn.Module):
  """q(\nu; \theta) is the prior on the auxiliary latent variables \nu."""
  def __init__(self,
               latent_size,
               flow_depth,
               hidden_size,
               hidden_degrees,
               activation,
               reverse,
               flow_std):
    super().__init__()
    modules = []
    self.latent_size = latent_size
    for flow_num in range(flow_depth):
      module = flow.AutoregressiveSampleAndLogProb(
        num_input=latent_size,
        use_context=False,
        hidden_size=hidden_size,
        hidden_degrees=hidden_degrees,
        activation=activation,
        flow_std=flow_std)
      modules.append(module)
      if reverse:
        modules.append(flow.Reverse(latent_size))
    self.q_nu = flow.FlowSequential(*modules)
    self.q_nu_0 = distributions.Normal(loc=0.0, scale=flow_std)

  def sample_base_distribution(self, num_samples):
    nu_0 = torch.randn((num_samples, self.latent_size), 
                       device=next(self.parameters()).device)
    return nu_0.mul_(self.q_nu_0.scale)

  def sample_and_log_prob(self, num_samples):
    nu_0 = self.sample_base_distribution(num_samples)
    log_q_nu_0 = self.q_nu_0.log_prob(nu_0).sum(-1)
    nu, log_q_nu = self.q_nu(nu_0)
    nu = reshape_lattice(nu)
    return nu, log_q_nu_0 + log_q_nu


class RealNVPPrior(nn.Module):
  """q(\nu; \theta) is the prior on the auxiliary latent variables \nu."""
  def __init__(self,
               latent_shape,
               flow_depth,
               hidden_size,
               flow_std):
    super().__init__()
    modules = [flow.CheckerSplit(latent_shape)]
    for flow_num in range(flow_depth):
      modules.append(flow.RealNVPPermuteSampleAndLogProb(
        in_channels=1,
        hidden_size=hidden_size,
        # invert mask opposite to prior 
        parity=True if flow_num % 2 == 1 else False))
    modules.append(flow.CheckerConcat(latent_shape))      
    self.q_nu = flow.RealNVPSequential(*modules)
    self.q_nu_0 = distributions.Normal(loc=0.0, scale=flow_std)
    self.latent_shape = latent_shape

  def sample_base_distribution(self, num_samples):
    nu_0 = torch.randn((num_samples,) + self.latent_shape, 
                       device=next(self.parameters()).device)
    return nu_0.mul_(self.q_nu_0.scale)

  def sample_and_log_prob(self, num_samples):
    nu_0 = self.sample_base_distribution(num_samples)
    log_q_nu_0 = self.q_nu_0.log_prob(nu_0).sum((1, 2))
    nu, log_q_nu = self.q_nu(nu_0)
    return nu, log_q_nu_0 + log_q_nu
