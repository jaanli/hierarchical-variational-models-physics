import torch
import torch.nn as nn
import numpy as np


class NormalLogProb(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, loc, scale, z):
    var = torch.pow(scale, 2)
    return -0.5 * torch.log(2 * np.pi * var) - torch.pow(loc - z, 2) / (2 * var)


class StandardNormalLogProb(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, z):
    return -0.5 * np.log(2 * np.pi) - 0.5 * torch.pow(z, 2)

  
