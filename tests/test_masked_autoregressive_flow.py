"""Debug IAF and MAF by fitting to the moons dataset."""
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import math
import yaml
import nomen
import random

import network
import moons
import flow

config = """
batch_size: 100
test_batch_size: 1000
learning_rate: 0.0001
flow_type: realnvp # masked_autoregressive
num_blocks: 5
num_hidden: 64
cuda: true
log_interval: 240  # one epoch for 24k datapoints with 100 batch size
max_iteration: 10000
out_dir: $LOG/debug
rho: 0.
use_tanh: true
seed: 582848
"""


def main(cfg):
  np.random.seed(cfg.seed)
  torch.manual_seed(cfg.seed)
  random.seed(48283)
  device = torch.device("cuda:0" if cfg.cuda else "cpu")
  dataset = moons.MOONS()
  kwargs = {'num_workers': 4, 'pin_memory': True} if cfg.cuda else {}
  train_tensor = torch.from_numpy(dataset.trn.x)
  train_dataset = torch.utils.data.TensorDataset(train_tensor)

  valid_tensor = torch.from_numpy(dataset.val.x)
  valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

  test_tensor = torch.from_numpy(dataset.tst.x)
  test_dataset = torch.utils.data.TensorDataset(test_tensor)

  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=cfg.batch_size, shuffle=True, **kwargs)

  valid_loader = torch.utils.data.DataLoader(
      valid_dataset,
      batch_size=cfg.test_batch_size,
      shuffle=False,
      drop_last=False,
      **kwargs)

  test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=cfg.test_batch_size,
      shuffle=False,
      drop_last=False,
      **kwargs)

  # plot real data
  fig, ax = plt.subplots()
  ax.plot(dataset.val.x[:,0], dataset.val.x[:,1], '.')
  ax.set_title('Real data')
  plt.savefig(cfg.out_dir / 'data.png')

  modules = []
  mask = network.mask.checkerboard((1, 2))
  base_mask = torch.from_numpy(mask)
  for flow_num in range(cfg.num_blocks):
    if cfg.flow_type == 'realnvp':
      if flow_num % 2 == 0:  # invert mask opposite to prior 
        mask = 1 - base_mask
      else:
        mask = base_mask
      modules.append(flow.RealNVPCoupling(input_shape=(1, 2),
                                                 hidden_size=10,
                                                 mask=mask,
                                                 kernel_size=1))
    elif cfg.flow_type == 'maf':
      modules.append(flow.MaskedAutoregressiveFlow(dataset.n_dims,
                                                   cfg.num_hidden,
                                                   use_context=False,
                                                   use_tanh=cfg.use_tanh))
      modules.append(flow.BatchNormalization(dataset.n_dims))
      modules.append(flow.Reverse(dataset.n_dims))
  model = flow.FlowSequential(*modules)

  # orthogonal initialization helps
  for module in model.modules():
    if isinstance(module, nn.Linear):
      nn.init.orthogonal_(module.weight)
      module.bias.data.fill_(0)
  model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), 
                               lr=cfg.learning_rate,
                               weight_decay=1e-6)

  p_u = torch.distributions.Normal(loc=torch.zeros(dataset.n_dims, device=device),
                                   scale=torch.ones(dataset.n_dims, device=device))

  train_iter = iter(train_loader)
  for step in range(cfg.max_iteration):
    try:
      data = next(train_iter)
    except StopIteration:
      train_iter = iter(train_loader)
      data = next(train_iter)
    data = data[0].to(device)
    optimizer.zero_grad()
    log_prob = log_prob_fn(model, p_u, data).sum() / data.shape[0]
    loss = -log_prob
    loss.backward()
    optimizer.step()
    if step % cfg.log_interval == 0:
      if np.isnan(loss.item()):
        raise ValueError("Loss hit nan!")
      print(f'epoch: {step * cfg.batch_size // len(train_dataset)}')
      print(f"step: {step}\tlog_lik: {log_prob.item():.2f}")
      for module in model.modules():
        if isinstance(module, flow.BatchNormalization):
          module.momentum = 0.

      # initialize the moving averages with the full dataset
      all_data = train_loader.dataset.tensors[0].to(data.device)
      with torch.no_grad():
        model(all_data)

      # update momentum for proper eval
      for module in model.modules():
        if isinstance(module, flow.BatchNormalization):
          module.momentum = 1.0
      valid_log_lik = evaluate(model, p_u, log_prob_fn, valid_loader, device)
      print(f"\tvalid log-lik: {valid_log_lik:.10f}")
      model.eval()
      plot(step, cfg.out_dir, model, dataset, device)
      model.train()


def log_prob_fn(model, p_u, data):
  u, log_prob_model = model(data)
  # u is shape (batch_size, latent_size)
  log_prob_u = torch.sum(p_u.log_prob(u), dim=-1, keepdim=True)
  return log_prob_u + log_prob_model.sum(-1, keepdim=True)


def evaluate(model, p_u, log_prob_fn, loader, device):
  model.eval()
  total_log_prob = 0.0
  for data in loader:
    data = data[0].to(device)
    with torch.no_grad():
      log_prob = log_prob_fn(model, p_u, data)
      total_log_prob += log_prob.sum().item()
  return total_log_prob / len(loader.dataset)


def plot(step, out_dir, model, dataset, device):
  # generate some examples
  np_x = np.linspace(-1.1, 2.1, 50).astype(np.float32)
  np_y = np.linspace(-0.75, 1.25, 50).astype(np.float32)
  xx, yy = np.meshgrid(np_x, np_y)
  np_u = np.stack([xx.flatten(), yy.flatten()]).T
  u_tens = torch.from_numpy(np_u).to(device)
  _, log_prob = model(u_tens)
  log_prob = log_prob.sum(-1).detach().cpu().numpy()
  log_prob_grid = log_prob.reshape((len(xx), len(xx)))
  xlabels = ['%.1f' % x for x in np_x]
  ylabels = ['%.1f' % x for x in np_y]
  ax = sns.heatmap(log_prob_grid[::-1], xticklabels=xlabels, yticklabels=ylabels[::-1])
  ax.set_title('Heatmap of log likelihood of the flow')
  name = "log-lik_maf_%d.png" % step
  plt.savefig(out_dir / name)
  plt.close()

if __name__ == "__main__":
  cfg = nomen.Config(dictionary=yaml.load(config))
  main(cfg)
