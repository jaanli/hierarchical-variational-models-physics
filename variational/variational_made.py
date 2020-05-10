import torch
from torch import nn
from network import made

class VariationalMADE(nn.Module):
  def __init__(self, latent_size, hidden_size):
    super().__init__()
    self.latent_size = latent_size
    self.net = made.MADE(num_input=latent_size, 
                         num_output=latent_size, 
                         num_hidden=hidden_size, 
                         use_context=False)
    # bernoulli log prob is equivalent to negative binary cross entropy
    self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

  @torch.no_grad()
  def sample(self, num_samples):
    sample = torch.zeros((num_samples, self.latent_size), 
                         device=next(self.parameters()).device)
    for i in range(self.latent_size):
      logits = self.net(sample)
      sample[:, i] = torch.bernoulli(torch.sigmoid(logits[:, i]))
    return sample

  def log_prob(self, sample):
    logits = self.net(sample)
    return -self.bce_with_logits(logits, sample).sum(-1)

