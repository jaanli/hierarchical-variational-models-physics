import torch

from inference import control_variate


class MeanFieldELBO:
  def __init__(self, config, model, variational):
    if config.rao_blackwellize and config.marginalize:
      raise ValueError('Not implemented: rao blackwellization and marginalization')
    if config.control_variate:
      assert config.num_samples_grad > 2, 'num_eps_samples must be > 2 for denominator in scaling of control variate'
    self.config = config
    self.p_z = model
    self.q_z = variational

  def compute_grad_natural_parameters(self, annealing_temp, logit, z, log_r_i_nu):
    cfg = self.config
    score_q_z = self.q_z.grad_log_prob(logit=logit, value=z)
    log_q_z = self.q_z.log_prob(logit=logit, value=z)
    if cfg.marginalize:
      E_del_nu_elbo = self.compute_grad_marginal(annealing_temp, logit, z)
      z_i_terms = E_del_nu_elbo + log_r_i_nu
      E_del_nu_z_i_terms = E_del_nu_elbo + score_q_z * log_r_i_nu
    else:
      if cfg.rao_blackwellize:
        log_p_z = -cfg.beta * self.p_z.rao_blackwellized_energy(z)
      else:
        log_p_z = -cfg.beta * self.p_z.energy(z)  # (num_samples,)
        log_p_z = log_p_z.unsqueeze(-1).unsqueeze(-1)
      elbo_mean_field = log_p_z / annealing_temp - log_q_z
      z_i_terms = elbo_mean_field if log_r_i_nu is None else elbo_mean_field + log_r_i_nu
      E_del_nu_z_i_terms = score_q_z * z_i_terms

    if cfg.control_variate:
      a = control_variate.optimal_scale(cfg.num_samples_grad, E_del_nu_z_i_terms, score_q_z)
      # subtracting scaling before multiplying by control variate works better 
      # than subtracting a * score_q_z
      E_del_nu_z_i_terms = score_q_z * (z_i_terms - a)
    return E_del_nu_z_i_terms

  def compute_grad_marginal(self, annealing_temp, logit, z):
    cfg = self.config
    energy_up, energy_down = self.p_z.rao_blackwellized_energy_marginals(z)
    log_p_z_one = (-cfg.beta * energy_up.view(energy_up.size(0), -1) 
                   / annealing_temp)
    log_p_z_zero = (-cfg.beta * energy_down.view(energy_down.size(0), -1)
                    / annealing_temp)
    log_q_z_one, log_q_z_zero = self.q_z.log_prob_marginals(logit)
    q_z_one, q_z_zero = self.q_z.probabilities(logit)
    score_q_z_one, score_q_z_zero = self.q_z.grad_log_prob_marginals(logit)
    return (q_z_one * score_q_z_one * (log_p_z_one - log_q_z_one)
            + q_z_zero * score_q_z_zero * (log_p_z_zero - log_q_z_zero))
