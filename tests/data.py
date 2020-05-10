import os
import torch
import torch.utils.data
import numpy as np
import h5py

def load_binary_mnist(cfg, **kwcfg):
  f = h5py.File(os.path.join(os.environ['DAT'], 'binarized_mnist.hdf5'), 'r')
  x_train = f['train'][::]
  x_val = f['valid'][::]
  x_test = f['test'][::]
  train = torch.utils.data.TensorDataset(torch.from_numpy(x_train))
  train_loader = torch.utils.data.DataLoader(train, batch_size=cfg.batch_size, shuffle=True)
  validation = torch.utils.data.TensorDataset(torch.from_numpy(x_val))
  val_loader = torch.utils.data.DataLoader(validation, batch_size=cfg.test_batch_size, shuffle=False)
  test = torch.utils.data.TensorDataset(torch.from_numpy(x_test))
  test_loader = torch.utils.data.DataLoader(test, batch_size=cfg.test_batch_size, shuffle=True)
  return train_loader, val_loader, test_loader
