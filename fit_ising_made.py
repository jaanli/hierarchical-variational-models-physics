import numpy as np
import torch
import yaml
import time
import nomen
import time

import model
import variational
import objective
import log
import stats


config = """
## MODEL
# system size
system_size: 4
# periodic or free boundary condition
boundary_condition: periodic
beta: 1.0
## VARIATIONAL
variational: made
hidden_size: 256
## OPTIMIZATION
num_samples: 1024
# use the score function as a control variate with optimal scaling 
control_variate: false
# marginalize over the mean-field variables
marginalize: false
rao_blackwellize: false
rmsprop_centered: false
momentum: 0.9
max_iteration: 100000
use_gpu: true
learning_rate: 0.00001
## LOGGING
num_samples_print: 65536
num_samples_grad: 10
log_interval: 100
log_dir: $LOG/debug
train_dir: $TMPDIR
experiment_name: null
seed: 58283
"""

EPSILON = 1.0E-8

def fit(cfg):
  torch.manual_seed(cfg.seed)
  device = torch.device('cuda:0' if cfg.use_gpu else 'cpu')
  logger = log.get_file_console_logger(cfg.train_dir /  'train.log')
  latent_size = cfg.system_size * cfg.system_size
  if cfg.boundary_condition == 'free':
    p_z = model.IsingSquareLatticeFreeBoundary()
  elif cfg.boundary_condition == 'periodic':
    p_z = model.IsingSquareLatticePeriodicBoundary()
  if cfg.variational == 'made':
    q_z = variational.VariationalMADE(latent_size=latent_size,
                                      hidden_size=cfg.hidden_size)
  p_z.to(device)
  q_z.to(device)
  elbo = objective.ELBO(config=cfg,
                        model=p_z,
                        variational=q_z)

  # evaluate partition function
  t0 = time.time()
  if cfg.system_size < 5:
    Z = model.partition_function(cfg.system_size, p_z, cfg.beta)
    true_free_energy = -1 / cfg.beta * np.log(Z) / latent_size
    logger.info(f'true partition function per spin {Z / latent_size:.3e}')
    logger.info(f'log Z per latent: {np.log(Z) / latent_size:.5f}')
    logger.info(f'free energy per spin: {true_free_energy:.5f}')
    logger.info(f'time to evaluate partition function (min): {(time.time() - t0) / 60:.2f}')

  optimizer = torch.optim.RMSprop(q_z.parameters(),
                                  lr=cfg.learning_rate,
                                  momentum=cfg.momentum, 
                                  centered=cfg.rmsprop_centered)
  t0 = time.time()
  best_np_elbo = -np.inf
  for step in range(cfg.max_iteration):
    if step % cfg.log_interval == 0:
      np_elbo, np_elbo_std, log_string = elbo.compute_objective()
      logger.info(f'step {step}\thierarchical elbo per latent:\t{np_elbo / latent_size:.3f}')
      free_energy_bound = -1 / cfg.beta * np_elbo / latent_size
      free_energy_std = 1 / cfg.beta * np_elbo_std / latent_size
      with (cfg.train_dir / 'log.csv').open('a') as f:
        f.write(f'{step},{np_elbo},{np_elbo_std},{free_energy_bound},{free_energy_std}\n')
      logger.info(f'step {step}\tbound on free energy per latent (std):\t{free_energy_bound:.3f} ({free_energy_std:.3f})')
      if cfg.system_size < 5:
        logger.info(f'step {step}\ttrue free energy per latent:\t{true_free_energy:.3f}')
      logger.info(log_string)
      logger.info(f'\t\ttime per iteration:\t{(time.time() - t0) / cfg.log_interval}')
      t0 = time.time()
      if np_elbo >= best_np_elbo:
        states = {'q_z': q_z.state_dict(),
                  'optimizer': optimizer.state_dict()}
        torch.save(states, cfg.train_dir / 'best_state_dict')
      name_module_list = [('q_z', q_z)]
      grad_var, grad_mean_sq = stats.grad_var_and_mean_sq(cfg.num_samples_grad, 
                                                          name_module_list,
                                                          elbo.compute_grad)
      for key, _ in name_module_list:
        logger.info(f'\t{key}\tmean_sq_g0:\t{grad_mean_sq[key]:.3e}\tvar_g0:\t{grad_var[key]:.3e}')
    optimizer.zero_grad()
    elbo.compute_grad()
    optimizer.step()


if __name__ == "__main__":
  dictionary = yaml.load(config)
  cfg = nomen.Config(dictionary)
  cfg.parse_args()
  fit(cfg)
