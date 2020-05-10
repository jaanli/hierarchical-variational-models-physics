import model
import variational

def bbvi(cfg):
  """Fit mean field Poisson to toy posterior with black box variational inference."""
  torch.manual_seed(cfg.seed)
  device = torch.device('cuda:0' if cfg.use_gpu else 'cpu')
#  p_z = CorrelatedPosterior()
  p_z = GridPosterior(p=0.4)
#  q_z = variational.MeanFieldPoisson()
  q_z = variational.MeanFieldBernoulli()
  q_z.inv_rate = torch.nn.Parameter(10 * torch.ones(cfg.latent_size, device=device))
  optimizer = torch.optim.RMSprop([q_z.inv_rate],
                                  lr=cfg.learning_rate, 
                                  momentum=0.9)
                                  #centered=True)
  p_z.to(device)
  linspace = np.linspace(0, 20, 21, dtype=int).astype(np.float32)
  plot_log_prob(cfg, device, lambda z: p_z.log_prob(z), 'p(z) latent distribution', 'log_p_z.png', 
                linspace=linspace)
  for step in range(cfg.max_iteration):
    with torch.no_grad():
      z = q_z.sample()
      log_q_z = q_z.log_prob(z).sum(-1, keepdim=True)
      log_p_z = p_z.log_prob(z).sum(-1, keepdim=True)
      elbo = log_p_z - log_q_z
      score_q_z = q_z.grad_log_prob(z)
      q_z.inv_rate.grad = -(score_q_z * elbo).mean(0)
    optimizer.step()
    if step % cfg.log_interval == 0:
      print(f'step {step}\telbo: {elbo.sum().cpu().detach().numpy():.3f}')
      print(q_z.rate)
      with torch.no_grad():
        plot_log_prob(cfg, device, lambda z: q_z.log_prob(z), 'q(z) approximate latent distribution', 'log_q_mean-field_z_step=%d' % step,
                      linspace=linspace)

if __name__ == '__main__':
  bbvi(cfg)
