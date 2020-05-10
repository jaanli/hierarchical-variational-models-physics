import copy
import numpy as np
import torch


@torch.no_grad()
def update_stats(params, prev_params):
  """Compute norm of updates."""
  update_scales = []
  param_scales = []
  fractions = []
  for param, prev_param in zip(params, prev_params):
    update = param.data - prev_param
    update_scale = np.linalg.norm(update.numpy().ravel())
    param_scale = np.linalg.norm(param.data.numpy().ravel())
    update_scales.append(update_scale)
    param_scales.append(param_scale)
    fractions.append(update_scale / param_scale)
  return (np.array(x) for x in [update_scales, param_scales, fractions])


def compute_var_and_mean_sq(lst):
  """Compute variance and mean square of a list of samples."""
  num_samples = len(lst)
  mean = np.mean(lst, 0)
  # estimate variance
  var = np.sum([np.square(x - mean) for x in lst], 0) / (num_samples - 1)
  # estimate E[x^2]. cannot estimate E[x]^2 without bias
  square = np.mean([np.square(x) for x in lst], 0)
  for _ in range(var.ndim):
    var = var.mean(0)
    square = square.mean(0)
  return var, square


def grad_var_and_mean_sq(num_samples, modules, compute_grad_fn):
  params_dict = {name: list(module.parameters())
                     for name, module in modules.items()}
  grads = {name: [[None] * num_samples] * len(params) 
           for name, params in params_dict.items()}
  grad_var = copy.deepcopy(grads)
  grad_mean_sq = copy.deepcopy(grads)
  for sample_idx in range(num_samples):
    compute_grad_fn()
    for key, params in params_dict.items():
      for param_idx, p in enumerate(params):
        grads[key][param_idx][sample_idx] = p.grad.cpu().clone().numpy()
        p.grad.zero_()
    torch.cuda.empty_cache()
  for key, params in params_dict.items():
    for i in range(len(params)):
      var, mean_sq = compute_var_and_mean_sq(grads[key][i])
      grad_var[key][i] = var
      grad_mean_sq[key][i] = mean_sq
    grad_var[key] = np.mean(grad_var[key])
    grad_mean_sq[key] = np.mean(grad_mean_sq[key])
  return grad_var, grad_mean_sq


@torch.no_grad()
def probplot(step, q_nu):
  """Check whether latent variables follow a uniform distribution using a probability plot."""
  nu, _ = q_nu(num_samples=8192)
  prob = torch.sigmoid(nu).cpu().squeeze().numpy()
  sns.set_context('talk')
  for i in range(cfg.system_size ** 2):
    fig, ax = plt.subplots(figsize=(10 * 1.618, 10))
    scipy.stats.probplot(prob[:, i], dist='uniform', plot=ax)
    ax.set(xlabel='Uniform quantiles', ylabel='Ordered Bernoulli probabilities', title=f'Beta = {cfg.beta}')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.savefig(cfg.train_dir / f'step={step}_quantile-quantile-plot-latent_variable={i}.png')
    plt.close()
