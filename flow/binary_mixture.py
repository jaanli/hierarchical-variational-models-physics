from torch import nn

class BinaryMixtureTransform(nn.Module):
  """Transform eps ~ Uniform(0, 1); nu = Q(eps). Then p(nu) = p(eps)MoG(nu).

  As Q is the inverse CDF or quantile function of a mixture of Gaussians (MoG).
  The PDF of the transformed variable is thus
      log det(dQ^{-1}/dnu) = log det(dCDF/dnu) = log MoG
  """
  def __init__(self, latent_size, num_components):
    super().__init__()
    # init mixture components, last dimension corresponds to for z={0, 1}
    self.loc = nn.Parameter(torch.Tensor(latent_size, num_components, 2))
    self.scale_arg = nn.Parameter(torch.Tensor(latent_size, num_components, 2))
    # mixture weight 
    self.weight_logit = nn.Parameter(torch.Tensor(latent_size, num_components, 2))
    self._normal_log_prob = NormalLogProb()
    self._softplus = nn.Softplus()
    self._log_softmax = nn.LogSoftmax(dim=1)  # components are in dimension 1
    self._normal_cdf = NormalCDF()
    self._logit = Logit()
    self._log_grad_logit = LogGradLogit()
    self.reset_parameters()

  def reset_parameters(self):
    print("Reset mixture transform parameters.")
    with torch.no_grad():
      self.loc.normal_(0.0, 1.0)
      self.scale_arg.fill_(inv_softplus(1.0))
      # fill mixture weights to uniform
      self.weight_logit.fill_(1.0)

  def forward(self, nu, z):
    """Return inverse (epsilon) and log density of the transformed sample nu."""
    # calculate log density: sum of log grad of logit and log mixture of gaussians
    log_weight = self._log_softmax(self.weight_logit)
    scale = self._softplus(self.scale_arg)
    # calculate log (\sum_k \pi_k Normal(nu; loc, scale))
    normal_log_prob = self._normal_log_prob(self.loc.unsqueeze(0), 
                                            scale.unsqueeze(0), 
                                            nu.unsqueeze(-1).unsqueeze(-1))
    log_prob_mixture = torch.logsumexp(normal_log_prob + log_weight, dim=2)
    # invert the transform; return CDF(nu), which is F(nu) = \sum_k \pi_k \Phi_k(nu)
    normal_cdf = self._normal_cdf(self.loc.unsqueeze(0), 
                                  scale.unsqueeze(0), 
                                  nu.unsqueeze(-1).unsqueeze(-1))
    mixture_cdf = (log_weight.unsqueeze(0) + torch.log(normal_cdf + TINY)).logsumexp(dim=2).exp()
    log_prob_logit = self._log_grad_logit(mixture_cdf)
    log_prob = log_prob_logit + log_prob_mixture
    epsilon = self._logit(mixture_cdf)
    # pick the transform corresponding to z={0, 1}
    epsilon = mask_last_dim(epsilon, z)
    log_prob = mask_last_dim(log_prob, z)
    return epsilon, log_prob


def mask_last_dim(tensor, binary_mask):
  """Pick the elements of tensor in the last dimension according to binary_mask."""
  return tensor[..., 0] * binary_mask + tensor[..., 1] * (1 - binary_mask)


class NormalCDF(nn.Module):
  """Normal CDF is \Phi = 0.5 (1 + erf(1/sqrt(2 * scale) * (x - loc)))."""

  def __init__(self):
    super().__init__()

  def forward(self, loc, scale, value):
    return 0.5 * (1 + torch.erf((value - loc) * scale.reciprocal() / math.sqrt(2)))
    

class Logit(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, input):
    return torch.log(input) - torch.log(1 - input)
    

class LogGradLogit(nn.Module):
  """Gradient of logit = log p - log(1 - p) is 1/p + 1/(1 - p) = 1 / (p(1 - p))
     log grad = -log(y(1 - y)) = -log y - log(1 - y)"""
  def __init__(self):
    super().__init__()

  def forward(self, input):
    return -torch.log(input) - torch.log(1 - input)
