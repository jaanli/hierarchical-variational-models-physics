import torch
from torch import nn


class GridPosterior(nn.Module):
  """Log probability of a 2D posterior defined by a 2x2 grid with 0.5 - p on the diagonal."""
  def __init__(self, p):
    super().__init__()
    # use XOR to compute; off diagonal is p, on-diagonal is 0.5 - p
    self.register_buffer('prob', torch.Tensor([0.5 - p, p]))

  def log_prob(self, z):
    # compute XOR to index diagonal / diagonal terms with 0, 1
    a, b = torch.chunk(z, chunks=2, dim=-1)
    index = (a + b - 2 * a * b).long()
    return torch.log(self.prob[index])
