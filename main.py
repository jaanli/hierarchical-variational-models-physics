import argparse
import numpy as np
import torch
import time
import json
from torch.utils.tensorboard import SummaryWriter

import model
import variational
import inference
import network
import decay
import log
import importance_sampling
import arguments

from inference import proximity
from moving_averages import ExponentialMovingAverage


def get_memory():
  torch.cuda.synchronize()
  max_memory = torch.cuda.max_memory_allocated()
  memory = torch.cuda.memory_allocated() 
  return memory / 10**9, max_memory / 10**9


def train(cfg):
  torch.manual_seed(cfg.seed)
  device = torch.device('cuda:0' if cfg.use_gpu else 'cpu')
  logger = log.get_file_console_logger(cfg.log_dir /  'train.log')
  with open(cfg.log_dir / 'args.txt', 'w') as f:
    json.dump(arguments.args_to_string(cfg), f, indent=2)

  if cfg.model == 'sk':
    p_z = model.SherringtonKirkpatrick(cfg.num_spins)
  elif cfg.boundary == 'free':
    p_z = model.IsingSquareLatticeFreeBoundary()
  elif cfg.boundary == 'periodic':
    p_z = model.IsingSquareLatticePeriodicBoundary()
  
  L = int(np.sqrt(cfg.num_spins))
  latent_shape = (L, L)
  if cfg.flow_type == 'realnvp':
    q_nu = variational.RealNVPPrior(latent_shape=latent_shape,
                                    flow_depth=cfg.flow_depth,
                                    hidden_size=cfg.hidden_size,
                                    flow_std=cfg.prior_std)
    r_nu = variational.RealNVPPosterior(latent_shape=latent_shape,
                                        flow_depth=cfg.flow_depth,
                                        flow_std=cfg.posterior_std,
                                        hidden_size=cfg.hidden_size)
  elif cfg.flow_type == 'realnvp_full_conditioning':
    q_nu = variational.RealNVPPrior(latent_shape=latent_shape,
                                    flow_depth=cfg.flow_depth,
                                    hidden_size=cfg.hidden_size,
                                    flow_std=cfg.prior_std)
    r_nu = variational.RealNVPFullConditioningPosterior(latent_shape=latent_shape,
                                        flow_depth=cfg.flow_depth,
                                        flow_std=cfg.posterior_std,
                                        hidden_size=cfg.hidden_size)
  else:
    q_nu = variational.AutoregressivePrior(latent_size=cfg.num_spins,
                             flow_depth=cfg.flow_depth,
                             hidden_size=cfg.hidden_size,
                             hidden_degrees=cfg.hidden_degrees,
                             activation=cfg.activation,
                             reverse=cfg.reverse,
                             flow_std=cfg.prior_std)
    r_nu = variational.BinaryPosterior(latent_size=cfg.num_spins,
                      flow_depth=cfg.flow_depth,
                      flow_std=cfg.posterior_std,
                      hidden_size=cfg.hidden_size,
                      hidden_degrees=cfg.hidden_degrees,
                      reverse=cfg.reverse,
                      activation=cfg.activation)
  q_z = variational.MeanFieldBernoulli()

  p_z.to(device)
  q_nu.to(device)
  r_nu.to(device)

  if cfg.checkpoint is not None:
    checkpoint = torch.load(cfg.checkpoint)
    q_nu.load_state_dict(checkpoint['q_nu'])
    r_nu.load_state_dict(checkpoint['r_nu'])
    print(f'Loaded parameters from: {cfg.checkpoint}')
  
  elbo = inference.HierarchicalELBO(config=cfg,
                                    model=p_z,
                                    variational_likelihood=q_z,
                                    variational_prior=q_nu,
                                    variational_posterior=r_nu)
  free_energy_fn = lambda log_partition_fn: -1 / cfg.beta * log_partition_fn / cfg.num_spins
  q_params = list(q_nu.parameters())
  r_params = list(r_nu.parameters())
  optimizer = torch.optim.RMSprop(q_params + r_params,
                                  lr=cfg.learning_rate,
                                  momentum=cfg.momentum, 
                                  centered=False)
  writer = SummaryWriter(cfg.log_dir, flush_secs=10)
  params = list(filter(lambda p: p.requires_grad, q_params + r_params))
  nparams = int(sum([np.prod(p.shape) for p in params]))
  logger.info('Total number of trainable parameters: {}'.format(nparams))
  writer.add_scalar('num_params', nparams)

  modules = {'q_nu': q_nu, 'r_nu': r_nu}
  prev_params = {'q_nu': log.get_cpu_params(q_nu),
                 'r_nu': log.get_cpu_params(r_nu)}
      
  if cfg.proximity_constraint:
    term_dict = log.summarize_elbo_terms(writer=writer, 
                                         step=0, 
                                         num_samples_print=cfg.num_samples_print,
                                         batch_size=cfg.print_batch_size, 
                                         elbo=elbo)
    # this is a scalar estimate of log q nu
    log_q_nu = torch.Tensor([term_dict['log_q_nu/mean']]).to(device)
    log_r_nu = torch.Tensor([term_dict['log_r_nu/mean']]).to(device)
    init_elbo = np.mean(elbo.sample_objective(cfg.num_samples_print, cfg.print_batch_size))
    init_elbo = np.abs(init_elbo)
    exponential_decay = decay.ExponentialDecay(start_value=init_elbo * cfg.init_magnitude_scale,
                                               end_value=0.0,
                                               decay_rate=cfg.decay_rate,
                                               decay_steps=cfg.decay_steps)
    moving_average = ExponentialMovingAverage(decay=cfg.moving_average_decay,
                                              tensor_dict={'log_q_nu': log_q_nu, 
                                                           'log_r_nu': log_r_nu})
    proximity_loss = proximity.ProximityConstraint(moving_average=moving_average)
    exponential_decay.print_values(10)
    elbo.set_proximity_loss(proximity_loss)
    

  tic = time.time()
  for step in range(cfg.max_iteration):
    print('step ', step)
    if step % (cfg.log_interval - 1) == 0:
      log.update_prev_params(modules, prev_params)
    if step % cfg.log_interval == 0:
      compute_time = time.time() - tic
      free_energy = log.compute_free_energy(writer, 
                                            step, 
                                            cfg.num_samples_print, 
                                            cfg.print_batch_size,
                                            free_energy_fn, 
                                            elbo.sample_objective)
      logger.info(f'step {step}\tFree energy bound: {free_energy:.3f}')
      logger.info(f'\ttime per iteration:'
                  f'\t{compute_time / cfg.log_interval}')
      writer.add_scalar('time_per_iter', compute_time / cfg.log_interval, step)
      # log.compute_grad_stats(writer, 
      #                        step, 
      #                        num_samples=100,
      #                        modules=modules,
      #                        grad_fn=lambda: elbo.compute_grad(annealing_temp))
      # if step > 0:
      #   log.compute_update_stats(writer, step, modules, prev_params)
      res = log.summarize_elbo_terms(writer, 
                                     step, 
                                     cfg.num_samples_print // 2, 
                                     cfg.print_batch_size, 
                                     elbo)
      if cfg.proximity_constraint:
        nu, log_q_nu = elbo.q_nu.sample_and_log_prob(cfg.num_samples_grad)
        z = elbo.q_z.sample(logit=nu)
        _, log_r_i_nu, log_r_nu = elbo.r_nu.inverse_and_log_prob(nu, z)
        assert log_r_i_nu is None
        total_constraint = 0.0
        for name, tensor in {'log_q_nu': log_q_nu, 'log_r_nu': log_r_nu}.items():
          constraint = elbo.proximity_loss.compute_constraint(name, tensor).mean()
          writer.add_scalar(f'objective/constraint_{name}', constraint, step)
          total_constraint += constraint
        writer.add_scalar('objective/total_constraint', total_constraint, step)
      tic = time.time()
    optimizer.zero_grad()
    if cfg.proximity_constraint and step >= 0:
      elbo.proximity_loss.magnitude = exponential_decay.get_value(step)
    elbo.compute_grad()
    optimizer.step()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  arguments.add_model(parser)
  arguments.add_variational(parser)
  arguments.add_optim(parser)
  arguments.add_logging(parser)
  arguments.add_proximity(parser)
  cfg = parser.parse_args()
  train(cfg)
