"""Fit hierarchical variational model to a correlated toy posterior.

Convention: 
  - first dimension is z sample dimension, then is the nu
    sample dimension, then is the latent dimension.
"""
import numpy as np
import torch
import pandas as pd
import yaml
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')
from matplotlib.colors import ListedColormap
import time
import collections
import nomen
import copy

import model
import variational

config = """
latent_size: 2
grid_max: 1
hidden_size: 256
num_eps_samples: 512
num_eps_samples_plot: 4096
num_eps_samples_print: 1024
num_samples_grad: 100
control_variate: true
num_z_samples: 512
num_z_samples_print: 1024
compute_grad_manually: false
nu_flow_depth: 5
r_flow_depth: 5
plot_grid_size: 100
num_samples_plot: 512
use_gpu: true
learning_rate: 0.0001
log_interval: 100
max_iteration: 100000
train_dir: $LOG/debug
seed: 58283
"""


def fit_hvm(cfg):
  """Fit HVM to toy discrete, correlated posterior."""
  torch.manual_seed(cfg.seed)
  device = torch.device('cuda:0' if cfg.use_gpu else 'cpu')
  # p_z = model.CorrelatedPosterior()
  p_z = model.GridPosterior(p=0.4)
  q_nu = variational.VariationalPrior(latent_size=cfg.latent_size,
                                      hidden_size=cfg.hidden_size,
                                      flow_depth=cfg.nu_flow_depth)
  # q_z = variational.MeanFieldPoisson()
  q_z = variational.MeanFieldBernoulli()
  r_nu = variational.VariationalPosterior(latent_size=cfg.latent_size,
                                          hidden_size=cfg.hidden_size,
                                          flow_depth=cfg.r_flow_depth,
                                          num_samples=cfg.num_eps_samples)
  p_z.to(device)
  q_nu.to(device)
  r_nu.to(device)
  plot_log_prob(cfg, device, lambda z: p_z.log_prob(z), 
                title='p(z) latent distribution', fname='log_p_z.png', grid_max=cfg.grid_max)
  nu_linspace = np.linspace(-1.0, #inv_softplus(0.0001),
                            1.0, #inv_softplus(cfg.grid_max), 
                            cfg.grid_max + 1,
                            dtype=np.float32)

  @torch.no_grad()
  def log_q_z_fn(z):
    nu, log_q_nu = q_nu(num_samples=cfg.num_eps_samples_plot)
    q_z.set_inverse_param(nu)
    log_q_z = q_z.log_prob(z)
    # q_{HVM}(z) = \int q(nu) q(z | nu)
    # log q(z) \approxeq log 1/S \sum_{i=1}^S log q(nu_0^i) + log q(nu^i) + log q(z | nu^i)
    log_q_z = torch.logsumexp(log_q_nu + log_q_z, 0) - np.log(cfg.num_eps_samples_plot)
    return log_q_z

  @torch.no_grad()
  def log_r_nu_fn(nu, z):
    log_r_nu_0_given_z, log_r_nu = r_nu(nu, z)
    # average out the num_z_samples dimension
    log_r_nu_0_given_z = log_r_nu_0_given_z.mean(0)
    assert log_r_nu_0_given_z.shape == log_r_nu.shape
    return log_r_nu_0_given_z + log_r_nu

  @torch.no_grad()
  def log_softplus_r_nu_fn(nu, z):
    # log density of softplus(nu) with nu ~ r(nu | z)
    # d / dy of x(y) is 1 / (1 - exp(-y)), where y(x) is softplus(nu)
    y = r_nu.r_nu_0._softplus(nu)
    return log_r_nu_fn(y, z) - torch.log(1 - torch.exp(-y))

  optimizer = torch.optim.RMSprop(list(q_nu.parameters()) + list(r_nu.parameters()),
                                  lr=cfg.learning_rate, centered=True)
  
  for step in range(cfg.max_iteration):
    if step % cfg.log_interval == 0:
      hier_elbo = compute_hier_elbo(cfg, step, p_z, q_nu, q_z, r_nu)
      print(f'step {step}\thierarchical elbo: {hier_elbo.cpu().detach().numpy():.3f}')
      grad_var, grad_mean_sq = compute_grad_var_and_mean_sq(cfg, p_z, q_nu, q_z, r_nu)
      for key in ['q_nu', 'r_nu']:
        print(f'\t{key}\tmean_sq_g0:\t{grad_mean_sq[key]:.3e}\tvar_g0:\t{grad_var[key]:.3e}')
      # with torch.no_grad():
      #   make_plots(cfg, step, device, q_nu, log_q_z_fn, log_r_nu_fn, log_softplus_r_nu_fn, nu_linspace)
    optimizer.zero_grad()
    compute_grad(cfg.control_variate,
                 cfg.compute_grad_manually,
                 cfg.num_eps_samples, 
                 cfg.num_z_samples, 
                 p_z, q_nu, q_z, r_nu)
    optimizer.step()


@torch.no_grad()
def leave_one_out_sum(x, dim):
  return (x.sum(dim, keepdim=True) - x).squeeze(dim)
  

def compute_grad(control_variate, manual, num_eps_samples, num_z_samples, p_z, q_nu, q_z, r_nu):
  nu, log_q_nu = q_nu(num_samples=num_eps_samples)
  q_z.set_inverse_param(nu)
  z = q_z.sample(num_samples=num_z_samples)
  log_q_z = q_z.log_prob(z).detach()#.sum(-1, keepdim=True).detach()
  score_q_z = q_z.grad_log_prob(z)
  log_p_z = p_z.log_prob(z)#.sum(-1, keepdim=True)
  elbo_mean_field = log_p_z - log_q_z
  log_r_nu_0_given_z, log_r_nu = r_nu(nu, z)
  log_r_nu = log_r_nu_0_given_z + log_r_nu.unsqueeze(0)
  log_r_nu_0_given_z = log_r_nu_0_given_z.detach()
  if control_variate:
    # use leave-one-out estimator for the optimal control variate
    # Cov(f, h) / Var(h) where h is score_q_z, f is the gradient of the ELBO
    # then E[h] = 0, which simplifies the covariance and variance
    score_q_z_sq = score_q_z.pow(2)
    # denominator for sample variance is N - 1 (Bessel's correction)
    # we leave out a sample so it is N - 2
    cov = leave_one_out_sum((elbo_mean_field + log_r_nu_0_given_z) * score_q_z_sq, dim=1) / (num_z_samples - 2)
    var = leave_one_out_sum(score_q_z_sq, dim=1) / (num_z_samples - 2)
    E_del_nu_elbo_mean_field = (score_q_z * (elbo_mean_field + log_r_nu_0_given_z - cov / var)).mean(0)
  else:
    E_del_nu_elbo_mean_field = (score_q_z * elbo_mean_field).mean(0)
    E_c_log_r = (score_q_z * log_r_nu_0_given_z.detach()).mean(0)

  assert control_variate, "Only supports control variate as we include E[c log r(nu | z)] in E_del_elbo"
  if manual:
    E_del_nu_log_r_nu, = torch.autograd.grad(log_r_nu.mean(0).sum(), nu, retain_graph=True)
    nu.backward((E_del_nu_elbo_mean_field 
                 + E_del_nu_log_r_nu
               #  + E_c_log_r  # included in E_del_nu_elbo_mean_field
               ), retain_graph=True)
    (-log_q_nu.sum(-1).sum(0)).backward(retain_graph=True)
    for p in q_nu.parameters():
      p.requires_grad = False
    log_r_nu.sum(-1).mean(0).sum().backward()
    for p in q_nu.parameters():
      p.requires_grad = True
    # convert the \sum_j into 1 / N \sum_j to get the expectation over s(eps)
    # multiply by -1 to maximize the hierarchical elbo
    for p in q_nu.parameters():
      p.grad.div_(-num_eps_samples)
    for p in r_nu.parameters():
      p.grad.div_(-num_eps_samples)
  else:
    hier_elbo_pre_grad = (
       # yields del_theta nu(eps; theta) del_nu L_{MF}
       (nu * E_del_nu_elbo_mean_field).sum(-1)
       # yields E_q(z | nu) [del_theta nu(eps; theta) * del_nu log r(nu | z)]
       + log_r_nu.sum(-1).mean(0)
       # yields del_theta nu(eps; theta) E[\sum_i del_nu log q(z_i | nu) * log r_i(nu | z)]
       # included in E_del_nu_elbo_mean_field
       #       + (nu * E_c_log_r).sum(-1)  
       # yields del_theta nu(eps; theta) del_nu log q(nu; theta)
       - log_q_nu.sum(-1)
      ).mean(0)
    loss = -hier_elbo_pre_grad
    loss.backward()


def compute_grad_var_and_mean_sq(cfg, p_z, q_nu, q_z, r_nu):
  q_nu_params = list(q_nu.parameters())
  r_nu_params = list(r_nu.parameters())
  grads = {'q_nu': [[None] * cfg.num_samples_grad] * len(q_nu_params), 
           'r_nu': [[None] * cfg.num_samples_grad] * len(r_nu_params)}
  grad_var = copy.deepcopy(grads)
  grad_mean_sq = copy.deepcopy(grads)
  key_params = [('q_nu', q_nu_params), ('r_nu', r_nu_params)]
  for sample_idx in range(cfg.num_samples_grad):
    compute_grad(cfg.control_variate,
                 cfg.compute_grad_manually,
                 cfg.num_eps_samples,
                 cfg.num_z_samples, 
                 p_z, q_nu, q_z, r_nu)
    for key, params in key_params:
      for param_idx, p in enumerate(params):
        grads[key][param_idx][sample_idx] = p.grad.clone()
        p.grad.zero_()
  for key, params in key_params:
    for i in range(len(params)):
      var, mean_sq = compute_var_and_mean_sq(grads[key][i])
      grad_var[key][i] = var.cpu().numpy()
      grad_mean_sq[key][i] = mean_sq.cpu().numpy()
    grad_var[key] = np.mean(grad_var[key])
    grad_mean_sq[key] = np.mean(grad_mean_sq[key])
  return grad_var, grad_mean_sq
  
  
@torch.no_grad()
def compute_var_and_mean_sq(lst):
  """Compute variance and mean square of a list of samples."""
  num_samples = len(lst)
  tensor = torch.stack(lst)
  mean = torch.mean(tensor, 0, keepdim=True)
  # estimate variance
  var = (tensor - mean).pow(2).sum(0) / (num_samples - 1)
  # estimate E[x^2]. cannot estimate E[x]^2 without bias
  square = tensor.pow(2).mean(0)
  return var.mean(0).mean(0), square.mean(0).mean(0)

@torch.no_grad()
def compute_hier_elbo(cfg, step, p_z, q_nu, q_z, r_nu):
  nu, log_q_nu = q_nu(num_samples=cfg.num_eps_samples_print)
  q_z.set_inverse_param(nu)
  z = q_z.sample(num_samples=cfg.num_z_samples_print)
  # print empirical fractions
  # print('true probabilities of 2x2 p(z) grid:')
  # print('diagonal:')
  # print(p_z.prob[0])
  # print('off-diagonal:')
  # print(p_z.prob[1])
  # print('empirical probabilities from q(z | nu) with nu ~ q(nu):')
  a, b = z.chunk(chunks=2, dim=-1)
  index = a + b - 2 * a * b  
  frac = index.mean(0).mean(0)  # proportion of XOR 1's
  print('diagonal:')
  pprint((1 - frac) / 2)
  print('off-diagonal:')
  pprint(frac / 2)
  arr = z.reshape((-1, z.shape[-1])).cpu().numpy()
  tuples = [tuple(row) for row in arr]
  print('empirical counts:')
  counter = collections.Counter(tuples)
  print(counter)
  print('fractions:')
  print([(key, value / len(arr)) for key, value in counter.items()])
  #pprint(q_z._sigmoid(nu.mean(0)))
  # save correlation matrix of z samples
  plot_z_histogram(cfg, z, 
                   title='histogram of z ~ q(z | nu) with nu ~ q(nu)', 
                   fname='q_z_sample_correlation_step=%d.png' % step)

  log_q_z = q_z.log_prob(z).sum(-1, keepdim=True)
  log_p_z = p_z.log_prob(z).sum(-1, keepdim=True)
  log_r_nu_0_given_z, log_r_nu = r_nu(nu, z)
  log_r_nu = log_r_nu_0_given_z + log_r_nu.unsqueeze(0)
  lst = [('dim 1: sigmoid(nu)', q_z.prob.mean(0)[0]),
         ('dim 2: sigmoid(nu)', q_z.prob.mean(0)[1]),
         ('log_p_z', log_p_z.sum(-1).mean(0).mean(0)),
         ('log_q_z', log_q_z.sum(-1).mean(0).mean(0)),
         ('log_q_nu', log_q_nu.sum(-1).mean(0).mean(0)),
         ('log_r_nu', log_r_nu.sum(-1).mean(0).mean(0)),
         ('log_r_nu_0_given_z', log_r_nu_0_given_z.sum(-1).mean(0).mean(0))]
  print('\n'.join([f'\t{name}\t{x.detach().cpu().numpy():.3f}' for name, x in lst]))
  return ((log_p_z - log_q_z).sum(-1).mean(0)
           + log_r_nu.sum(-1).mean(0)
           - log_q_nu.sum(-1)).mean(0).squeeze()


def make_plots(cfg, step, device, q_nu, log_q_z_fn, log_r_nu_fn, log_softplus_r_nu_fn, nu_linspace):
  plot_log_prob(cfg, device, log_q_z_fn, 'q(z) approximate latent distribution', 'log_q_hvm_z_step=%d' % step, grid_max=cfg.grid_max)
  plot_q_nu_hist(cfg, device, q_nu, 'softplus(nu) with nu ~ q(nu) variational prior', 'softplus_q_nu_histogram_step=%d' % step,
                 linspace=np.log(1 + np.exp(nu_linspace)),
                 axes_transform=lambda x: np.log(1 + np.exp(x)))
  plot_q_nu_hist(cfg, device, q_nu, 'q(nu) variational prior', 'q_nu_histogram_step=%d' % step,
                 linspace=nu_linspace)
  plot_log_prob(cfg, device, log_r_nu_fn, 'r(nu | z) recursive variational posterior', 'log_r_nu_step=%d' % step, nu_linspace=nu_linspace, grid_max=cfg.grid_max)
  plot_log_prob(cfg, device, log_softplus_r_nu_fn, 
                title='log prob of softplus(nu) with nu ~ r(nu | z)',
                fname='log_softplus_r_nu_step=%d' % step,
                nu_linspace=nu_linspace,
                axes_transform=lambda x: np.log(1 + np.exp(x)),
                grid_max=cfg.grid_max)


def pprint(x): 
  arr = x.cpu().numpy()
  arr = arr[::-1]
  print(arr)


def inv_softplus(x):
  return np.log(np.exp(x) - 1)

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

def get_grid(linspace):
  xx, yy = np.meshgrid(linspace, linspace)
  return np.stack([xx.flatten(), yy.flatten()]).T


def plot_q_nu_hist(cfg, device, q_nu, title, fname, linspace, axes_transform=lambda x: x):
  with torch.no_grad():
    nu, _ = q_nu(num_samples=cfg.num_eps_samples_plot)
  np_nu = nu.cpu().numpy()
  fig, ax = plt.subplots(figsize=(12, 10))
  cmap = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=False)
  cmap = ListedColormap(cmap.as_hex())
  x = axes_transform(np_nu)
  linspace = axes_transform(linspace)
  ax.hist2d(x[:, 0], x[:, 1], bins=(linspace, linspace), cmap=cmap)
  ax.set(title=title, xlabel='dimension 1', ylabel='dimension 2')
  plt.savefig(cfg.train_dir / fname, bbox_inches="tight")
  plt.close()


def plot_log_prob(cfg, device, log_prob_fn, title, fname, nu_linspace=None, axes_transform=lambda x: x, grid_max=20):
  linspace = np.linspace(0, grid_max, grid_max + 1, dtype=int).astype(np.float32)
  np_z = get_grid(linspace)
  if nu_linspace is not None:
    np_nu = get_grid(nu_linspace)
    assert np_nu.shape == np_z.shape, 'to condition on z in r(nu | z), z, nu must be same shape'
  np_log_prob = np.zeros(np_z.shape[0])
  for row, np_z_row in enumerate(np_z):
    z = torch.from_numpy(np_z_row).to(device)
    # 3D tensor: (num_z_samples, num_nu_samples, latent_size)
    z = z.unsqueeze(0).unsqueeze(0)
    if nu_linspace is not None:
      nu = torch.from_numpy(np_nu[row]).to(device)
      # 2D tensor: (num_nu_samples, latent_size)
      nu = nu.unsqueeze(0)
      log_prob = log_prob_fn(nu, z)
    else:
      log_prob = log_prob_fn(z)
    np_log_prob[row] = np.sum(log_prob.cpu().numpy())
  grid = np_log_prob.reshape((len(linspace), len(linspace)))
  if nu_linspace is not None:
    labels = ['%.3f' % axes_transform(x) for x in nu_linspace]
  else:
    labels = ['%d' % axes_transform(x) for x in linspace]
  fig, ax = plt.subplots(figsize=(12, 10))
  cmap = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=False)
  # flip y axis, which is the first dimension (rows)
  grid = grid[::-1]
  kwargs = dict(xticklabels=labels, yticklabels=labels[::-1], cmap=cmap, linewidths=0.05)
  sns.heatmap(grid, ax=ax, **kwargs)
  ax.set(title=title, xlabel='dimension 1', ylabel='dimension 2')
  plt.savefig(cfg.train_dir / fname, bbox_inches='tight')
  plt.close()


def plot_z_histogram(cfg, var, title, fname):
  """For a variable, plot correlation in the last dimension."""
  size = var.shape[-1] + 1
  var = var.reshape((-1, var.shape[-1]))
  x = var.cpu().numpy()
  fig, ax = plt.subplots(figsize=(12, 10))
  cmap = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=False)
  cmap = ListedColormap(cmap.as_hex())
  linspace = np.linspace(0, var.shape[-1], var.shape[-1], dtype=int)
  ax.hist2d(x[:, 0], x[:, 1], bins=(linspace, linspace), cmap=cmap)
  ticks = list(range(1, size + 1))
  ax.set_xticks(ticks)
  ax.set_yticks(ticks)
  ax.set(title=title)
  plt.savefig(cfg.train_dir / fname, bbox_inches="tight")
  plt.close()

  # df = pd.DataFrame(data=var.cpu().numpy(),
  #                   columns=['%d' % x for x in range(1, size)])
  # corr = df.corr()
  # mask = np.zeros_like(corr, dtype=np.bool)
  # mask[np.triu_indices_from(mask)] = True
  # f, ax = plt.subplots(figsize=(11, 9))
  # cmap = sns.diverging_palette(10, 10, as_cmap=True)
  # sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
  #           square=True, linewidths=.5, cbar_kws={"shrink": .5})
  # plt.savefig(cfg.train_dir / fname, bbox_inches='tight')
  # plt.close()


def print_grads(module):
  lst = []
  for i, p in enumerate(module.parameters()):
    lst.append(p.grad.norm().detach().cpu().numpy())
  print(['%.3f' % x for x in lst])


if __name__ == "__main__":
  dictionary = yaml.load(config)
  cfg = nomen.Config(dictionary)
  cfg.parse_args()
  fit_hvm(cfg)
