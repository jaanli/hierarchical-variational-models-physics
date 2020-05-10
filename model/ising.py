import torch
from torch import nn
import numpy as np
import itertools
import scipy.integrate

from util import reshape_lattice

def ising_exact_free_energy(beta, J_horizontal, J_vertical):
  """Calculate exact free energy per site.

  https://en.wikipedia.org/wiki/Square-lattice_Ising_model
  """
  K = beta * J_horizontal
  L = beta * J_vertical
  cosh2Kcosh2L = np.cosh(2 * K) * np.cosh(2 * L)
  k = 1 / (np.sinh(2 * K) * np.sinh(2 * L))
  def theta_integrand(theta):
    """Integrand in expression for free energy of square lattice."""
    return np.log(cosh2Kcosh2L +
                  1 / k * np.sqrt(1 + k ** 2 - 2 * k * np.cos(2 * theta)))
  integral, _ = scipy.integrate.quad(theta_integrand, 0, np.pi)
  F = np.log(2) / 2 + 1 / (2 * np.pi) * integral
  return -F / beta


class IsingSquareLatticeFreeBoundary(nn.Module):
  """Square-lattice 2D Ising model with Rao-Blackwellization.

  The Hamiltonian (energy) is H(s) = -\sum_{<ij>} s_i s_j

  For a system size of L x L spins, we compute the unnormalized log
  probability (Hamiltonian) as an L x L tensor where each entry a_{ij}
  contains terms in the Hamiltonian with the $ij$-th spin.
  """
  def __init__(self):
    super().__init__()

  def energy(self, z):
    """Compute energy of a sample configuration, shape (num_samples,)."""
    spins = 2 * z - 1
    nearest_down = spins[..., :-1, :] * spins[..., 1:, :]
    nearest_right = spins[..., :-1] * spins[..., 1:]
    return -(nearest_down.sum((1, 2)) + nearest_right.sum((1, 2)))

  def rao_blackwellized_energy(self, z):
    """Return (L, L) tensor that only has terms in the energy involving spin at site (i, j)."""
    spins = 2 * z - 1
    nearest_below = spins.roll(shifts=1, dims=1)
    terms_below = spins * nearest_below
    # spins in last row do not interact with spins in the first row
    terms_below[..., -1, :].fill_(0)
    terms_above = terms_below.roll(shifts=-1, dims=1)
    # spins in first row do not interact with spins in the last row
    terms_above[..., 0, :].fill_(0)
    nearest_right = spins.roll(shifts=1, dims=2)
    terms_right = spins * nearest_right
    # spins in last column do not interact with spins in first column
    terms_right[..., -1].fill_(0)
    terms_left = terms_right.roll(shifts=-1, dims=2)
    # spins in first column do not interact with spins in last column
    terms_left[..., 0].fill_(0)
    return -(terms_above + terms_below + terms_left + terms_right)


class IsingSquareLatticePeriodicBoundary(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def energy(self, z):
    """Return scalar value of the energy."""
    spins = 2 * z - 1
    energy = 0
    for dim in [1, 2]:
      energy += (spins * spins.roll(shifts=1, dims=dim)).sum((1, 2))
    return -energy

  def rao_blackwellized_energy(self, z):
    """Compute terms in the energy only involving spin (i,j).

    Returns tensor of shape (num_samples, L, L) where L is lattice length.
    """
    spins = 2 * z - 1
    energy_terms_with_ij = torch.zeros_like(spins)
    for dim in [1, 2]:
      energy_terms = spins * spins.roll(shifts=1, dims=dim)
      energy_terms_with_ij = energy_terms_with_ij + energy_terms + energy_terms.roll(shifts=-1, dims=dim)
    return -energy_terms_with_ij

  def rao_blackwellized_energy_marginals(self, spins):
    """Terms in the energy only involving spin (i,j), marginalizing out every (i,j) spin."""
    spins_up = torch.ones(1, spins.shape[1], spins.shape[2], device=spins.device)
    spins_down = -spins_up 
    res = []
    for state in [spins_up, spins_down]:
      energy_terms_with_ij = torch.zeros_like(spins)
      for dim in [1, 2]:
        energy_terms = state * spins.roll(shifts=1, dims=dim)
        energy_terms_with_ij = energy_terms_with_ij + energy_terms + energy_terms.roll(shifts=-1, dims=dim)
      res.append(-energy_terms_with_ij)
    return res
