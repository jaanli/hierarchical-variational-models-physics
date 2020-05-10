import math
import torch
from torch import nn
from torch.nn import init


class InvertibleNetwork(nn.Module):
  """Invertible neural network. Implements differentiable inverse for use in flows."""
  def __init__(self, latent_size, negative_slope):
    super().__init__()
    self.f = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    self.f_inv = InverseLeakyReLU(
        negative_slope=self.f.negative_slope, inplace=True)
    self.log_grad_f_inv = LogGradInverseLeakyReLU()
    self.weight = nn.Parameter(torch.Tensor(3, latent_size))
    self.bias = nn.Parameter(torch.Tensor(3, latent_size))
    self.reset_parameters()

  def reset_parameters(self):  
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    bound = 0.01
    init.uniform_(self.bias, -bound, bound)
    
  def forward(self, input):
    """Input is of shape (batch_size, latent_dim)."""
    h1 = self.f(self.weight[0] * input + self.bias[0])
    h2 = self.f(self.weight[1] * h1 + self.bias[1])
    return self.weight[2] * h2 + self.bias[2]

  def inverse(self, nu):
    sum_log_weight_inv = self.weight.abs().log().sum(dim=0)
    weight_inv = self.weight.reciprocal()
    w_inv_nu_minus_b = weight_inv[2] * (nu - self.bias[2])
    h2_inv = weight_inv[1] * self.f_inv(w_inv_nu_minus_b) - self.bias[1]
    return weight_inv[0] * self.f_inv(h2_inv) - self.bias[0]

  def log_det_grad_inverse(self, nu):
    """Log absolute value of determinant of Jacobian of inverse transform."""
    sum_log_weight_inv = self.weight.abs().log().sum(dim=0)
    weight_inv = self.weight.reciprocal()
    w_inv_nu_minus_b = weight_inv[2] * (nu - self.bias[2])
    #import ipdb; ipdb.set_trace()
    return (sum_log_weight_inv + 
            self.log_grad_f_inv(w_inv_nu_minus_b) +
            self.log_grad_f_inv(weight_inv[1] * self.f_inv(w_inv_nu_minus_b) - self.bias[1]))

  
class InverseLeakyReLU(nn.Module):
  """LeakyReLU^{-1}(y) = {y if y >= 0 else 1/negative_slope * y"""
  def __init__(self, negative_slope=1e-2, inplace=False):
    super().__init__()
    self.negative_slope_inverse = 1 / negative_slope 
    self.inplace = inplace

  def forward(self, input):
    return F.leaky_relu(input, self.negative_slope_inverse, self.inplace)

  
class LogGradInverseLeakyReLU(nn.Module):
  """\partial_y LeakyReLU^{-1}(y) = {1 if y >= 0 else 1 / negative_slope."""
  def __init__(self, negative_slope=1e-2, inplace=False):
    super().__init__()
    self.log_negative_slope_inverse = -math.log(negative_slope)
    self.inplace = inplace

  def forward(self, input):
    """Calculate log of derivative of inverse."""
    mask = input >= 0
    return (1 - mask).float() * self.log_negative_slope_inverse
