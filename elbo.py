import torch
from inference import control_variate


class ELBO:
  def __init__(self, config, model, variational):
    self.config = config
    self.p_z = model
    self.q_z = variational

  @torch.no_grad()
  def compute_objective(self):
    cfg = self.config
    z = self.q_z.sample(cfg.num_samples_print)
    spins = 2 * z - 1
    spins = self.p_z.reshape_lattice(spins)
    log_p_z = self.p_z.energy(spins)
    log_q_z = self.q_z.log_prob(z)
    elbo = log_p_z - log_q_z
    return elbo.mean().item(), elbo.std().item(), ''

  def compute_grad(self):
    cfg = self.config
    z = self.q_z.sample(cfg.num_samples)
    log_q_z = self.q_z.log_prob(z)
    spins = 2 * z - 1
    spins = self.p_z.reshape_lattice(spins)
    log_p_z = self.p_z.energy(spins)
    with torch.no_grad():
      elbo = log_p_z - log_q_z
    # todo: this is incorrect - different samples are mixed, not leave-one-out!
    loss = -((elbo - elbo.mean()) * log_q_z).mean()
    loss.backward()
    
    
    
    
