import math
import torch
from torch import nn


class SherringtonKirkpatrick(nn.Module):
  """Sherrington-Kirkpatrick model, fully-connected, with Gaussian interactions."""
  def __init__(self, num_spins):
    super().__init__()
    # J_{ij} ~ Gaussian(0, 1 / \sqrt{N})
    J = torch.randn([num_spins, num_spins]) / math.sqrt(num_spins)
    # spins cannot be connected to themselves; set J_{ii} = 0 with zero diagonal
    J = torch.triu(J, diagonal=1)
    # construct symmetric matrix J_{ij} = J_{ji}
    J += J.t()
    self.register_buffer('J', J)

  def energy(self, spins):
    """Compute Hamiltonian given [num_samples, num_spins] tensor.

    Divide by two to fix double-counting.
    """
    num_samples = spins.shape[0]
    spins = spins.view(num_samples, -1)
    # (num_samples, num_spins) x (num_spins, num_spins) -> (num_samples, num_spins)
    # this is computed in spins @ self.J
    return ((spins @ self.J) * spins).sum(-1) / 2
    
