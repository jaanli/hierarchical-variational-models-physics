import torch

EPSILON = 1.0E-8


@torch.no_grad()
def optimal_scale(num_eps_samples, grad, score_q_z):
  # Following https://github.com/blei-lab/deep-exponential-families/blob/master/bbvi.cpp
  # use leave-one-out estimator for the optimal control variate
  # then E[score_q_z] = 0 simplifies the covariance and variance
  cov = leave_one_out_sum(grad * score_q_z, dim=0)
  # denominator for sample variance is N - 1 (Bessel's correction)
  # we leave out a sample so it is N - 2
  cov /= (num_eps_samples - 2)
  var_score = leave_one_out_sum(score_q_z.pow(2), dim=0) / (num_eps_samples - 2)
  # sampling low-variance score functions is probable; prevent division by zero
  return cov / (var_score + EPSILON)


@torch.no_grad()
def leave_one_out_sum(x, dim):
  return (x.sum(dim, keepdim=True) - x).squeeze(dim)
