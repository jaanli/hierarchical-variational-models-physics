import torch
from torch import nn


class FlowSequential(nn.Sequential):
  """Forward pass with log determinant of the Jacobian."""
  def forward(self, input, context=None):
    total_log_prob = torch.zeros(input.size(0), device=input.device)
    for block in self._modules.values():
      input, log_prob = block(input, context)
      total_log_prob += log_prob
    return input, total_log_prob

  def inverse(self, input, context=None):
    total_log_prob = torch.zeros(input.size(0), device=input.device)
    for block in reversed(self._modules.values()):
      input, log_prob = block.inverse(input, context)
      total_log_prob += log_prob
    return input, total_log_prob


def get_memory():
  torch.cuda.synchronize()
  max_memory = torch.cuda.max_memory_allocated()
  memory = torch.cuda.memory_allocated() 
  return memory / 10**9, max_memory / 10**9


class RealNVPSequential(nn.Sequential):
  """Assumes first and last module are CheckerSplit and CheckerUnsplit."""

  def forward(self, input, context=None):
    total_log_prob = torch.zeros(input.size(0), device=input.device)
    modules = list(self._modules.values())
    split = modules.pop(0)
    concat = modules.pop()
    transf, const = split(input)
    for module in modules:
      transf, const, log_prob = module(transf, const, context)
      total_log_prob += log_prob
    return concat(transf, const), total_log_prob

  def inverse(self, input, context=None):
    total_log_prob = torch.zeros(input.size(0), device=input.device)
    modules = list(self._modules.values())
    split = modules.pop(0)
    concat = modules.pop()
    transf, const = split(input)
    for module in reversed(modules):
      transf, const, log_prob = module.inverse(transf, const, context)
      total_log_prob += log_prob
    return concat(transf, const), total_log_prob


class SplitSequential(nn.Sequential):
  """Assumes first and last module are CheckerSplit and CheckerConcat."""

  def forward(self, transf, const, context=None):
    total_log_prob = torch.zeros(transf.size(0), device=transf.device)
    for module in self._modules.values():
      transf, const, log_prob = module(transf, const, context)
      total_log_prob += log_prob
    return transf, const, total_log_prob

  def inverse(self, transf, const, context=None):
    total_log_prob = torch.zeros(transf.size(0), device=transf.device)
    for module in reversed(self._modules.values()):
      transf, const, log_prob = module.inverse(transf, const, context)
      total_log_prob += log_prob
    return transf, const, total_log_prob

    
