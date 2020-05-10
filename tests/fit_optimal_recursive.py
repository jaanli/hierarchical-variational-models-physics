import torch
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')
import matplotlib.pyplot as plt
import numpy as np
import addict
import pathlib

import variational
import network

torch.set_anomaly_enabled(True)

cfg = addict.Dict(latent_size=1,
                  num_samples=100000,
                  learning_rate=1e-4,
                  momentum=0.9,
                  seed=242329,
                  log_interval=10,
                  max_iteration=10000,
                  train_dir=pathlib.Path('/Users/jaan/tmp/debug'))


def plot(module, z, ylabel, path):
  np_nu = np.linspace(-10, 10, 1000, dtype=np.float32)
  nu = torch.from_numpy(np_nu)
  np_values = module.log_prob(nu, z).numpy()
  fig, ax = plt.subplots(figsize=(5 * 1.618, 5))
  ax.plot(np_nu, np_values)
  ax.set(xlabel=r'$\nu$', ylabel=ylabel)
  plt.tight_layout()
  plt.savefig(path, bbox_inches='tight')
  plt.close()


def fit(cfg):
  z = torch.ones(1)
  p_nu = variational.OptimalRecursiveApproximation()
  plot(p_nu, z, ylabel=r'$\log q_{POST}(\nu \mid z) \propto$', path=cfg.train_dir / 'log_p_nu.png')

  r_nu = variational.VariationalRecursiveApproximation(latent_size=cfg.latent_size)
  optimizer = torch.optim.RMSprop(r_nu.parameters(),
                                  lr=cfg.learning_rate,
                                  momentum=cfg.momentum)


  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)

  # todo: switch to learning two transforms, for z=1, 0
  for step in range(cfg.max_iteration):
    nu, log_r_nu = r_nu(cfg.num_samples, z)
    log_p_nu = p_nu.log_prob(nu, z)
    kl = log_r_nu - log_p_nu
    # average over samples, sum over latent dimensions
    loss = kl.mean(dim=0).sum(dim=0)
    if step % cfg.log_interval == 0:
      print(loss, nu.mean())
      with torch.no_grad():
        plot(r_nu,
             z,
             ylabel=r'$\log r(\nu \mid z)$',
             path=cfg.train_dir / ('step=%d_log_r_nu.png' % step))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
      
if __name__ == '__main__':
  fit(cfg)
