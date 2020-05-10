import torch
from torch import nn


class CorrelatedPoissonPosterior(nn.Module):

  """Log probability of a 2D posterior that is a mixture of three Poisson distributions."""
  def __init__(self):
    super().__init__()
    big_1 = 13
    big_2 = 9
    self.register_buffer('first', torch.Tensor([0.01, 0.01]))
    self.register_buffer('second', torch.Tensor([0.01, big_2]))
    self.register_buffer('third', torch.Tensor([big_1, 0.01]))
    self.register_buffer('fourth', torch.Tensor([big_1, big_2]))

  def log_prob(self, z):
    p_list = [dist.Poisson(rate) for rate in [self.first, self.second, self.third, self.fourth]]
    exp_terms = [p.log_prob(z) - math.log(4) for p in p_list]
    return torch.logsumexp(torch.stack(exp_terms), dim=0)
