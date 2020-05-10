"""Mean-field variational likelihoods q(z | nu)."""
import torch
import torch.nn as nn


class MeanFieldPoisson(nn.Module):
  """Mean field Poisson family of variational distributions."""
  def __init__(self):
    super().__init__()
    self._softplus = nn.Softplus()
    self._sigmoid = nn.Sigmoid()
    self.inverse_rate = None

  @property
  def rate(self):
    return self._softplus(self.inverse_rate)

  def set_inverse_param(self, inverse_rate):
    self.inverse_rate = inverse_rate
    self.distribution = torch.distributions.Poisson(self.rate)

  def sample(self, num_samples):
    return self.distribution.sample((num_samples,))
    
  def log_prob(self, value):
    return self.distribution.log_prob(value)

  @torch.no_grad()
  def grad_log_prob(self, value):
    """Evaluate gradient of the log probability or score function.
    
    \nabla_\nu \log p(k; softplus(\nu)) = sigmoid(-\nu) (k / softplus(\nu)  - 1)
    """
    return self._sigmoid(-self.inverse_rate) * (value / self.rate - 1.0)
    

class MeanFieldBernoulli(nn.Module):
  def __init__(self):
    super().__init__()
    self._sigmoid = nn.Sigmoid()
    self._softplus = nn.Softplus()
    self._log_sigmoid = nn.LogSigmoid()
    self.logit = None
    # bernoulli log prob is equivalent to negative binary cross entropy
    self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
    
  @torch.no_grad()
  def sample(self, logit):
    return torch.bernoulli(self._sigmoid(logit))

  @torch.no_grad()
  def log_prob(self, logit, value):
    logit, value = torch.broadcast_tensors(logit, value)
    return -self.bce_with_logits(logit, value)

  @torch.no_grad()
  def log_prob_marginals(self, logit):
    """Log probability, marginalized: log p(z; pi) = z log pi + (1-z) log (1 - pi).
    
    pi = sigmoid(nu), so log (1 - pi) = -softplus(nu)
    """
    return self._log_sigmoid(logit), -self._softplus(logit)

  def probabilities(self, logit):
    # p(z = 0) = 1 - sigmoid(logit) = sigmoid(-logit)
    return self._sigmoid(logit), self._sigmoid(-logit)

  @torch.no_grad()
  def grad_log_prob(self, logit, value):
    """Gradient of log probability, score function.

    \nabla_\nu \log p(z; sigmoid(\nu)) = {1 - sigmoid(nu) if z == 1 
                                          else -sigmoid(nu)}
                                       = {sigmoid(-nu) if z == 1 
                                          else -sigmoid(nu)
    """
    boolean = (value > 0).float()
    return boolean * self._sigmoid(-logit) + (1 - boolean) * (-self._sigmoid(logit))

  @torch.no_grad()
  def grad_log_prob_marginals(self, logit):
    """Gradient of log probability, marginalized."""
    return self._sigmoid(-logit), -self._sigmoid(logit)
