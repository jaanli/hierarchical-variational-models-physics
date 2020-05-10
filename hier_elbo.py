import numpy as np
import collections
import torch
from tqdm import tqdm

from .mean_field_elbo import MeanFieldELBO


class HierarchicalELBO(MeanFieldELBO):
  def __init__(self, 
               config,
               model, 
               variational_likelihood, 
               variational_prior, 
               variational_posterior,
               proximity_loss=None
    ):
    super().__init__(config, model, variational_likelihood)
    self.config = config
    self.p_z = model
    self.q_z = variational_likelihood
    self.q_nu = variational_prior
    self.r_nu = variational_posterior
    self.proximity_loss = None

  def set_proximity_loss(self, proximity_loss):
    self.proximity_loss = proximity_loss

  @torch.no_grad()
  def sample_objective(self, num_samples, batch_size=2**13):
    cfg = self.config
    res = []
    assert num_samples >= batch_size
    for _ in tqdm(range(num_samples // batch_size)):
      hier_elbo = self.sample_objective_single_batch(batch_size)
      res.extend(hier_elbo.tolist())
    return res
  
  @torch.no_grad()
  def sample_objective_single_batch(self, batch_size, return_terms=False):
    nu, log_q_nu = self.q_nu.sample_and_log_prob(num_samples=batch_size)
    z = self.q_z.sample(logit=nu)
    nu_0, log_r_i_nu, log_r_nu = self.r_nu.inverse_and_log_prob(nu, z)
    if log_r_i_nu is not None:
      dims = tuple(range(1, log_r_i_nu.ndim))
      log_r_nu = log_r_i_nu.sum(dims) + log_r_nu
    log_q_z = self.q_z.log_prob(logit=nu, value=z).sum((1, 2))
    log_p_z = -self.config.beta * self.p_z.energy(z)
    hier_elbo = (log_p_z - log_q_z + log_r_nu - log_q_nu).squeeze()
    if return_terms:
      res = {'log_p_z': log_p_z,
             'log_q_z_entropy': -log_q_z, 
             'log_r_nu': log_r_nu, 
             'log_q_nu': log_q_nu,
             'nu': nu,
             'nu_0': nu_0}
      return {k: v.cpu().numpy() for k, v in res.items()}
    else:
      return hier_elbo.cpu().detach().numpy()
    
  def compute_grad(self, annealing_temp=1.0):
    cfg = self.config
    nu, log_q_nu = self.q_nu.sample_and_log_prob(cfg.num_samples_grad)
    z = self.q_z.sample(logit=nu)
    _, log_r_i_nu, log_r_nu = self.r_nu.inverse_and_log_prob(nu, z)
    dims = tuple(range(1, log_r_nu.ndim))
    if log_r_i_nu is not None:
      log_r_nu = log_r_i_nu.sum(dims) + log_r_nu  # (num_samples,)
      log_r_i_nu = log_r_i_nu.detach()  # (num_samples, L, L)
    E_del_nu_z_i_terms = super().compute_grad_natural_parameters(
        logit=nu, 
        z=z,
        log_r_i_nu=log_r_i_nu,
        annealing_temp=annealing_temp)
    hier_elbo_pre_grad = (
         # yields del_theta nu(eps; theta) del_nu L_{MF}
         # both have 2 lattice dimensions; sum over these
         (nu * E_del_nu_z_i_terms).sum(dims)
         # yields E_q(z | nu) [del_theta nu(eps; theta) * del_nu log r(nu | z)]
         + log_r_nu / annealing_temp
         # yields del_theta nu(eps; theta) del_nu log q(nu; theta)
         - log_q_nu
        )  # do not take expectation over s(epsilon), only after adding proximity loss
    # loss is negative elbo
    loss = -hier_elbo_pre_grad
    if self.proximity_loss is not None:
      tensor_dict = {'log_q_nu': log_q_nu, 'log_r_nu': log_r_nu}
      loss += self.proximity_loss.compute_total_constraint(tensor_dict)
      self.proximity_loss.moving_average.update(tensor_dict)
    # take expectation over s(epsilon) and compute gradients
    loss.mean(0).backward()
