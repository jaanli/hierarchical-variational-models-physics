import torch


class ProximityConstraint:
  def __init__(self, moving_average):
    self.magnitude = 0.0
    self.moving_average = moving_average

  def compute_total_constraint(self, tensor_dict):
    constraint = 0.0
    for name, tensor in tensor_dict.items():
      constraint += self.compute_constraint(name, tensor)
    return constraint

  def compute_constraint(self, name, tensor):
    diff = self.moving_average.state[name] - tensor
    abs_diff = diff.abs()
    indicator = (abs_diff < 1).float()
    sq_diff = 0.5 * diff.pow(2) + 0.5
    return self.magnitude * (indicator * abs_diff + (1 - indicator) * sq_diff)

  def compute_grad(self, tensors):
    """Inverse Huber loss is defined as:

    |x - y| if |x - y| < 1, and 0.5(x - y)^2 + 0.5 otherwise.
    
    The derivative w.r.t. x is:
    
    sgn(x - y) if |x - y| < 1, and (x - y) otherwise.
    """
    constraint = 0.0
    for tensor, avg in zip(tensors, self.moving_average):
      diff = avg - tensor
      indicator = (diff < 1).float()
      sgn = diff.sign()
      p.grad += self.magnitude * (indicator * sgn + (1 - indicator) * diff)      
      

