import numpy as np
import scipy.integrate

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

for beta in [0.4, 0.5]:
  print(beta, ising_exact_free_energy(beta, 1, 1))
