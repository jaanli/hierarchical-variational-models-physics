import logging
import numpy as np
import collections

import stats
import importance_sampling

def get_file_console_logger(filename):
  filename = str(filename)
  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s',
                      datefmt='%m-%d %H:%M',
                      filename=filename,
                      filemode='a')
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logger = logging.getLogger('')
  # only add the console handler if there is only the file handler
  if len(logger.handlers) == 1:
    logger.addHandler(console)
  return logger


def print_grads(module):
  lst = []
  for i, p in enumerate(module.parameters()):
    lst.append(p.grad.norm().detach().cpu().numpy())
  print(['%.3f' % x for x in lst])


def compute_free_energy(writer, step, num_samples, batch_size, free_energy_fn, objective_fn):
  elbo_samples = objective_fn(num_samples, batch_size)
  # free energy using elbo
  elbo_mean = np.mean(elbo_samples)
  elbo_std = np.std(elbo_samples)
  elbo_free_energy = free_energy_fn(elbo_mean)
  elbo_free_energy_std = np.abs(free_energy_fn(elbo_std))
  writer.add_scalar('objective/elbo', elbo_mean)
  writer.add_scalar('free_energy/elbo', elbo_free_energy, step)
  writer.add_scalar('free_energy/elbo_std', elbo_free_energy_std, step)
  # free energy using importance sampling
  log_partition_fn = importance_sampling.logsumexp_mean(elbo_samples)
  free_energy = free_energy_fn(log_partition_fn)
  writer.add_scalar('free_energy/importance_sampling', free_energy, step)
  # free energy using importance sampling bound
  k_list = [2, 8]
  for k in k_list:
    log_partition_fn = importance_sampling.log_partition_bound(elbo_samples, k=k)
    free_energy = free_energy_fn(log_partition_fn)
    writer.add_scalar(f'free_energy/importance_bound_k{k}', free_energy, step)
  return free_energy


def compute_grad_stats(writer, step, num_samples, modules, grad_fn):
  grad_var, grad_mean_sq = stats.grad_var_and_mean_sq(num_samples, 
                                                      modules, 
                                                      grad_fn)
  for key in modules:
    writer.add_scalar(f'{key}/grad_var', grad_var[key], step)
    writer.add_scalar(f'{key}/grad_mean_sq', grad_mean_sq[key], step)


def compute_update_stats(writer, step, modules, prev_params):
  for name, module in modules.items():
    update_scales, param_scales, fractions = stats.update_stats(
      get_cpu_params(module), prev_params[name])
    writer.add_scalar(
      f'update/{name}_mean_update_scale', np.mean(update_scales), step)
    writer.add_scalar(
      f'update/{name}_mean_param_scale', np.mean(param_scales), step)
    writer.add_scalar(
      f'update/{name}_mean_update_to_param_scale', np.mean(fractions), step)
    writer.add_histogram(
      f'update/{name}_update_to_param_scale', fractions, step)
    writer.add_histogram(
      f'update/{name}_update_scale', update_scales, step)
    writer.add_histogram(
      f'update/{name}_param_scale', param_scales, step)
    


def update_prev_params(modules, prev_params):
  for name, module in modules.items():
    prev_params[name] = [p.data.cpu() for p in module.parameters()]


def get_cpu_params(module):
  return [p.data.cpu() for p in module.parameters()]


def summarize_elbo_terms(writer, step, num_samples_print, batch_size, elbo):
  dct = collections.defaultdict(lambda: [])
  for _ in range(num_samples_print // batch_size):
    term_dict = elbo.sample_objective_single_batch(
        batch_size, return_terms=True)
    for key, np_arr in term_dict.items():
      dct[key].append(np_arr)
  res = {}
  for key, lst in dct.items():
    if len(lst[0].shape) == 0:
      arr = np.array(lst)
    else:
      arr = np.concatenate(lst)
    writer.add_scalar(f'{key}/mean', np.mean(arr), step)
    writer.add_scalar(f'{key}/std', np.std(arr), step)
    writer.add_scalar(f'{key}/min', np.min(arr), step)
    writer.add_scalar(f'{key}/max', np.max(arr), step)
    print(f'{key}/mean', np.mean(arr), step)
    res[f'{key}/mean'] = np.mean(arr)
    print(f'{key}/std', np.std(arr), step)
    print(f'{key}/min', np.min(arr), step)
    print(f'{key}/max', np.max(arr), step)
  writer.add_histogram('nu/hist', np.concatenate(dct['nu']).mean(0), step)
  writer.add_histogram('r_nu_nu_0/hist', np.concatenate(dct['nu_0']).mean(0), step)
  return res
