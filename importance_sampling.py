from scipy.special import logsumexp
import numpy as np
import torch
import collections
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')
import scipy.stats 


def logsumexp_mean(samples):
  """Hierarchical importance sampling to estimate the partition function.

  log \int f(z) \approx = \log 1/K \sum_i f(z_i) r(\nu_i | z_i) / q(z_i, \nu_i)

  where f(z) = \exp (-\beta H(z)) is the Boltzmann factor for Ising model.

  Compute this using log-sum-exp for numerical stability.

  Input:
    - samples: list of samples of hierarchical ELBO

  Output:
    - Estimate of the partition function using importance sampling.
  """
  K = len(samples)
  return logsumexp(samples) - np.log(K)


def log_partition_bound(samples, k):
  """Hierarchical importance sampling lower bound on partition function.

  log \int f(z) >= E_q [\log 1/K \sum_i f(z_i) r(\nu_i | z_i) / q(z_i, \nu_i) ]

  This strictly improves with number of inner samples K.
  """
  total_samples = len(samples)
  assert total_samples % k == 0, 'Total samples must be divisible by K'
  num_outer_samples = total_samples // k
  samples = np.array(samples).reshape(num_outer_samples, k)
  integrand = logsumexp(samples, axis=1, b=1 / k)
  # compute outer expectation
  return logsumexp(integrand) - np.log(num_outer_samples)


def uniform_proposal(cfg, num_samples, p_z):
  """Importance sampling with uniform proposal."""
  res = []
  device = next(p_z.parameters()).device
  for _ in range(num_samples // 8192):
    spins = torch.rand(8192, cfg.system_size, cfg.system_size, device=device).round().mul_(2).add_(-1)
    energy = p_z.energy(spins)
    log_p_z = -cfg.beta * energy
    log_q_z = np.log(0.5) * (cfg.system_size ** 2)
    elbo = log_p_z - log_q_z
    res.extend(elbo.cpu().numpy().tolist())
  return res


def compute_std(sample_list, sample_mean):
  var = 0.
  for sample in sample_list:
    var += (sample - sample_mean)**2
  var /= len(sample_list) - 1
  return np.sqrt(var)


def plot_bounds(num_samples_list, df, fname, true_free_energy):
  fig, ax = plt.subplots(figsize=(10 * 1.618, 10))
  for key in df:
    ax.plot(num_samples_list, df[key], label=key)
  ax.axhline(y=true_free_energy, linestyle='--', label='Analytic')
  ax.set_xscale('log')
  ax.legend()
  ax.set(xlabel='Number of samples', ylabel='Free energy')
  plt.savefig(fname)
  plt.close()


def plot_errors(num_samples_list, df, fname, true_free_energy):
  fig, ax = plt.subplots(figsize=(10 * 1.618, 10))
  for key in df:
    relative_error = np.abs((true_free_energy - df[key]) / true_free_energy)
    ax.plot(num_samples_list, relative_error, label=key)
  ax.axhline(y=true_free_energy, linestyle='--', label='Analytic')
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.legend()
  ax.set(xlabel='Number of samples', ylabel='Relative error in free energy')
  plt.savefig(fname)
  plt.close()


def mixture_proposal_elbo_samples(num_samples, hier_elbo, checkpoints):
  """Use a mixture of q(nu) distributions as proposals, at many temperatures.
  
  q(nu) = \sum_k \pi_k q_k(nu). 

  To sample, draw k ~ Categorical(1/K), then draw nu ~ q_k(nu).
  """
  raise ValueError('Incorrect: does not include q_k(nu) for all k in proposal densities!')
  component_samples = []
  num_components = len(checkpoints)
  for checkpoint in tqdm(checkpoints):
    checkpoint = torch.load(checkpoint)
    hier_elbo.q_nu.load_state_dict(checkpoint['q_nu'])
    res = hier_elbo.compute_objective(num_samples)
    component_samples.append(res['hier_elbo'])
  component_samples = np.array(component_samples)
  mixture_components = np.random.choice(num_components, size=num_samples, replace=True)
  return component_samples[mixture_components, np.arange(num_samples)]
