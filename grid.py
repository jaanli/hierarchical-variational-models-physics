import copy
import jobs
import collections
import pathlib
import numpy as np


def get_slurm_script_gpu(log_dir, command):
  """Returns contents of SLURM script for a gpu job."""
  return """#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:tesla_p100:1
#SBATCH --mem=32GB
#SBATCH --output={}/slurm_%j.out
#SBATCH -t 5:59:00
module load anaconda3/2019.10 cudatoolkit/10.1 cudnn/cuda-10.1/7.6.3
conda activate dev
{}
""".format(log_dir, command)


if __name__ == '__main__':
  # use -u to force print statements to be unbuffered (print statements are synchronous)
  np.random.seed(42329)
  commands = ["python -u main.py"]

  ## global options
  grid = collections.OrderedDict()
  log_dir = pathlib.Path(pathlib.os.environ['LOG']) / 'vi-for-physics' 
  grid['seed'] = 58283
  grid['model'] = ['sk']
  grid['boundary'] = ['periodic']
  grid['max_iteration'] = [1000000000]
  grid['use_gpu'] = [True]
  grid['num_samples_grad'] = [1024]
  grid['flow_depth'] = [6]
  grid['activation'] = ['relu'] #prelu
  grid['num_samples_print'] = 8192
  grid['variational_posterior'] = ['RealNVPPosterior']
  grid['hidden_degrees'] = ['equal']
  grid['prior_std'] = [1.0] #np.random.uniform(0.05, 0.5, size=3)
  grid['posterior_std'] = 1.0 # was 5.0
  grid['reverse'] = [False]
  grid['control_variate'] = [True]
  grid['rao_blackwellize'] = [True]
  grid['marginalize'] = [False]
  grid['learning_rate'] = [1e-5]
  grid['momentum'] = [0.9]
  grid['log_interval'] = [1000]
  grid['beta'] = [0.4]

  experiment = 'pvi-ising'
  grid['flow_type'] = 'realnvp_full_conditioning'
  grid['model'] = 'ising'
#  grid['num_spins'] = [L**2 for L in [16, 256, 512]]
  grid['hidden_size'] = [8]
  grid['num_samples_print'] = 2**26
  grid['print_batch_size'] = 2**16
  grid['log_interval'] = 5000

  #---------------ising + pvi
  grid = copy.deepcopy(grid)
  grid['model'] = 'ising'
  grid['num_spins'] = 16 ** 2
  grid['num_samples_print'] = 2 ** 18
  grid['print_batch_size'] = 2 ** 13 # 128
  grid['num_samples_grad'] = 1024

  grid['proximity_constraint'] = True  
  grid['init_magnitude_scale'] = 1.0
  grid['decay_steps'] = 10000
  grid['decay_rate'] = [1e-5, 1e-10, 1e-20, 1e-30, 1e-40]
  grid['moving_average_decay'] = 0.9999
  
  grid['log_interval'] = 1000
  dir_keys = jobs.get_dir_keys(grid)
  dir_keys.insert(0, 'num_spins')
  for cfg in jobs.param_grid(grid):
    cfg['log_dir'] = jobs.make_log_dir(cfg, log_dir, experiment, dir_keys)
    jobs.submit(commands, cfg, get_slurm_script_gpu)

  # grid = copy.deepcopy(grid)
  # grid['num_spins'] = [512 ** 2]
  # grid['num_samples_print'] = 2**20
  # grid['print_batch_size'] = 2**13
  # grid['num_samples_grad'] = 2 ** 6
  # grid['log_interval'] = 500
  # dir_keys = jobs.get_dir_keys(grid)
  # dir_keys.insert(0, 'num_spins')
  # for cfg in jobs.param_grid(grid):
  #   cfg['log_dir'] = jobs.make_log_dir(cfg, log_dir, experiment, dir_keys)
  #   jobs.submit(commands, cfg, get_slurm_script_gpu)

  # grid = copy.deepcopy(grid)
  # grid['num_spins'] = [256 * 256]
  # grid['num_samples_print'] = 2**20
  # grid['print_batch_size'] = 2**13
  # grid['log_interval'] = 500
  # dir_keys = jobs.get_dir_keys(grid)
  # dir_keys.insert(0, 'num_spins')
  # for cfg in jobs.param_grid(grid):
  #   cfg['log_dir'] = jobs.make_log_dir(cfg, log_dir, experiment, dir_keys)
  #   jobs.submit(commands, cfg, get_slurm_script_gpu)

  # # ----------------- SK model
  # grid = copy.deepcopy(grid)
  # grid['experiment'] = 'sk-residualmade-sigmoid-arg-bias'
  # grid['model'] = ['sk']
  # grid['num_spins'] = [20]
  # grid['num_samples_print'] = 2**23
  # grid['print_batch_size'] = 2**15
  # grid['log_interval'] = 102391
  # dir_keys = jobs.get_dir_keys(grid)
  # dir_keys.insert(0, 'num_spins')
  # for cfg in jobs.param_grid(grid):
  #   cfg['log_dir'] = jobs.make_log_dir(cfg, dir_keys)
  #   jobs.submit(commands, cfg, get_slurm_script_gpu)


  # grid['model'] = 'sk'
  # grid['rao_blackwellize'] = [False] # not possible for sk

  # grid = copy.deepcopy(grid)
  # grid['num_spins'] = [4096]
  # grid['num_samples_print'] = 2 ** 18
  # grid['print_batch_size'] = 2 ** 10
  # grid['num_samples_grad'] = 2 ** 5
  # grid['log_interval'] = 5000
  # dir_keys = jobs.get_dir_keys(grid)
  # dir_keys.insert(0, 'num_spins')
  # for cfg in jobs.param_grid(grid):
  #   cfg['log_dir'] = jobs.make_log_dir(cfg, log_dir, experiment, dir_keys)  
  #   jobs.submit(commands, cfg, get_slurm_script_gpu)

  # grid = copy.deepcopy(grid)
  # grid['num_spins'] = [16384] # 128 ** 2
  # grid['num_samples_print'] = 2 ** 17
  # grid['print_batch_size'] = 2 ** 10
  # grid['num_samples_grad'] = 2 ** 4
  # grid['log_interval'] = 1000
  # dir_keys = jobs.get_dir_keys(grid)
  # dir_keys.insert(0, 'num_spins')
  # for cfg in jobs.param_grid(grid):
  #   cfg['log_dir'] = jobs.make_log_dir(cfg, log_dir, experiment, dir_keys)
  #   jobs.submit(commands, cfg, get_slurm_script_gpu)
