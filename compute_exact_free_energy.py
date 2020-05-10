import torch
import model
import inference

if __name__ == '__main__':
  model_name = 'sk'
  num_spins = 20
  use_gpu = True
  beta = 0.4
  free_energy_fn = lambda log_partition_fn: -1 / beta * log_partition_fn / num_spins

  device = torch.device('cuda:0' if use_gpu else 'cpu')

  if model_name == 'sk':
    p_z = model.SherringtonKirkpatrick(num_spins)
  elif boundary == 'free':
    p_z = model.IsingSquareLatticeFreeBoundary()
  elif boundary == 'periodic':
    p_z = model.IsingSquareLatticePeriodicBoundary()
  p_z.to(device)

  log_Z = inference.exact_log_partition(num_spins, 
                                      p_z, 
                                      beta, 
                                      device, 
                                      reshape=True if model_name == 'ising' else False)

  print(free_energy_fn(log_Z))
