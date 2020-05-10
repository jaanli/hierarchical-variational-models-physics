import torch


class ExponentialMovingAverage:
  def __init__(self, decay, tensor_dict):
    self.decay = decay
    self.state = {key: x.detach().clone() for key, x in tensor_dict.items()}

  @torch.no_grad()
  def update(self, tensor_dict):
    for name, tensor in tensor_dict.items():
      # take average of tensor over samples
      self.state[name].mul_(self.decay).add_(1 - self.decay, tensor.mean(0))    
