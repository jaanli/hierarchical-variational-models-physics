import numpy as np


def reshape_lattice(tensor):
  """Reshape a tensor to be on a square lattice."""
  L = int(np.sqrt(tensor.shape[-1]))  # system size
  return tensor.view(tensor.shape[0], L, L)


