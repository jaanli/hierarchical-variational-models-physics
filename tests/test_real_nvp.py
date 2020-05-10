import pytest
import numpy as np
import torch
import network
import flow    
import variational
import time
import itertools
from torch import nn

torch.manual_seed(2)

use_gpu = True

device = torch.device('cuda:0' if use_gpu else 'cpu')


@pytest.fixture(params=[4])
def L(request):
  return request.param


def conv2d_coupling(L):
  mask = get_mask(L)
  coupling = flow.RealNVPInverseAndLogProb(
      hidden_size=16,
      kernel_size=3,
      mask=mask)
  return coupling.to(device)


def get_mask(L):
  return torch.from_numpy(network.mask.checkerboard((L, L))).to(device)


def to_numpy(tensor):
  return tensor.cpu().numpy().astype(np.float32)


def _get_memory():
  torch.cuda.synchronize()
  max_memory = torch.cuda.max_memory_allocated()
  memory = torch.cuda.memory_allocated() 
  return memory / 10**9, max_memory / 10**9


def fast_coupling():
  coupling = flow.RealNVPFastInverseAndLogProb(hidden_size=16)
  return coupling.to(device)


def perm_coupling(L, parity=False, num_blocks=1):
  modules = [flow.CheckerSplit((L, L))]
  for _ in range(num_blocks):
    modules.append(flow.RealNVPPermuteInverseAndLogProb(in_channels=1, 
                                                        hidden_size=16, 
                                                        parity=parity))
  modules.append(flow.CheckerConcat((L, L)))
  net = flow.RealNVPSequential(*modules)
  return net.to(device)


def _test_speeds(L, num_blocks=1):
  print('\n')
  conv2d = flow.FlowSequential(*[conv2d_coupling(L) for _ in range(num_blocks)])
  t0 = time.time()
  _test_coupling(conv2d, L)
  print(f'L: {L}\t slow:\t{time.time() - t0:.3f} s')
  fast = flow.FlowSequential(*[fast_coupling() for _ in range(num_blocks)])
  t0 = time.time()
  _test_coupling(fast, L)
  print(f'L: {L}\tfast:\t{time.time() - t0:.3f} s')
  perm = perm_coupling(L, num_blocks)
  t0 = time.time()
  _test_coupling(fast, L)
  print(f'L: {L}\tfast:\t{time.time() - t0:.3f} s')



def _test_memory_conv(L, num_blocks=1):
  p = torch.distributions.Normal(0, 1)
  x = p.sample((L, L)).to(device)
  x = x.unsqueeze(0)
  m0, max0 = _get_memory()
  net = flow.FlowSequential(*[conv2d_coupling(L) for _ in range(num_blocks)])
  m1, max1 = _get_memory()
  print('init mem, max:', m1 - m0, max1 - max0)
  y, log_x = net(x)
  m2, max2 = _get_memory()
  print('fwd mem, max:', m2 - m1, max2 - max1)

def _test_memory_fast(L, num_blocks):
  p = torch.distributions.Normal(0, 1)
  x = p.sample((L, L)).to(device)
  x = x.unsqueeze(0)
  m0, max0 = _get_memory()
  net = flow.FlowSequential(*[fast_coupling() for _ in range(num_blocks)])
  m1, max1 = _get_memory()
  print('init mem, max:', m1 - m0, max1 - max0)
  y, log_x = net(x)
  m2, max2 = _get_memory()
  print('fwd mem, max:', m2 - m1, max2 - max1)


def _test_memory_perm(L, num_blocks):
  p = torch.distributions.Normal(0, 1)
  x = p.sample((L, L)).to(device)
  x = x.unsqueeze(0)
  m0, max0 = _get_memory()
  modules = [flow.CheckerSplit((L, L))]
  for _ in range(num_blocks):
    modules.append(flow.RealNVPPermuteInverseAndLogProb(in_channels=1, hidden_size=16))
  modules.append(flow.CheckerConcat((L, L)))  
  net = flow.RealNVPSequential(*modules).to(device)
  m1, max1 = _get_memory()
  print('init mem, max:', m1 - m0, max1 - max0)
  y, log_x = net(x)
  m2, max2 = _get_memory()
  print('fwd mem, max:', m2 - m1, max2 - max1)


def _test_coupling(coupling, L):
  p = torch.distributions.Normal(0, 1)
  x = p.sample((L, L)).to(device)
  x = x.unsqueeze(0)
  y, log_x = coupling(x)
  x_pred, log_x_pred = coupling.inverse(y)
  #print(x_pred, '\n', x)
  assert torch.allclose(x_pred, x, rtol=0.01)
  #print(log_x_pred, log_x)
  assert torch.allclose(log_x_pred, log_x, rtol=0.01)


def test_fast_parity(L):
  coupling = flow.RealNVPFastInverseAndLogProb(hidden_size=16, parity=True)
  coupling.to(device)
  _test_coupling(coupling, L)


def test_perm_parity(L):
  _test_coupling(perm_coupling(L, parity=True), L)


def test_prior():
  q_nu = variational.RealNVPPrior(latent_shape=(4, 4),
                                  flow_depth=6,
                                  hidden_size=16,
                                  flow_std=1.0)

  nu_0 = q_nu.sample_base_distribution(1)
  log_q_nu_0 = q_nu.q_nu_0.log_prob(nu_0).sum((1, 2))
  nu, log_q_nu = q_nu.q_nu(nu_0)
  nu_0_pred, log_q_nu_pred = q_nu.q_nu.inverse(nu)
  log_q_nu_0_pred = q_nu.q_nu_0.log_prob(nu_0_pred).sum((1, 2))
  for pred, actual in [(nu_0_pred, nu_0),
                       (log_q_nu_pred, log_q_nu),
                       (log_q_nu_0_pred, log_q_nu_0)]:
    print(pred, actual)
    assert torch.allclose(pred, actual)


def test_posterior():
  r_nu = variational.RealNVPPosterior(latent_shape=(4, 4),
                                      flow_depth=6,
                                      hidden_size=16,
                                      flow_std=1.0)
  nu = torch.randn(1, 4, 4)
  z = torch.round(torch.rand(1, 4, 4))
  nu_0, log_r_nu_0, log_r_nu = r_nu.inverse_and_log_prob(nu, z)
  nu_pred, log_r_nu_pred = r_nu.r_nu.inverse(nu_0)
  for pred, actual in [(nu_pred, nu),
                       (log_r_nu_pred, log_r_nu)]:
    print(pred, actual)
    assert torch.allclose(pred, actual)


def get_jacobian(net, x, noutputs):
  """From https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa"""
  x = x.squeeze()
  n = x.size()[0]
  x = x.repeat(noutputs, 1)
  x.requires_grad_(True)
  y = net(x)
  y.backward(torch.eye(noutputs))
  return x.grad.data


def test_rectangle_shapes(L):
  net = network.Conv2dRect(in_channels=1, hidden_size=16, inner_activation='leaky_relu', final_activation=None)
  x = torch.round(torch.rand(1, L, L // 2)) * 2 - 1
  out = net(x)
  print(L, out.shape)
  assert tuple(out.shape) == (1, L, L // 2)



def get_jacobian(net, x, noutputs):
  """From https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa"""
  x = x.squeeze()
  n = x.size()[0]
  x = x.repeat(noutputs, 1)
  x.requires_grad_(True)
  y, _ = net(x)
  y.backward(torch.eye(noutputs))
  return x.grad.data


def _test_jacobian(coupling, L):
  mask = get_mask(L)
  # invert the mask
  np_mask = (~to_numpy(mask).astype(bool)).astype(np.float32)
  p = torch.distributions.Normal(0, 1)
  x = p.sample((L, L)).to(device)
  x = x.unsqueeze(0)
  x.requires_grad = True
  y, log_x = coupling(x)
  J = np.zeros((L ** 2, L ** 2))
  y = y.squeeze().flatten()
  for i in range(L ** 2): 
    for j in range(L ** 2):
      y[i].backward(retain_graph=True)
      J[i, j] = x.grad.flatten()[j].item()
      x.grad.zero_()
  log_det_J = np.log(np.abs(np.linalg.det(J)))
  print(log_det_J, log_x)
  #assert np.allclose(log_det_J, log_x.item())
  input_vars = np.where(J.sum(1) == 1)[0]
  # realnvp only takes half of the input variables
  assert len(input_vars) == L ** 2 // 2
  # other half of the variables depend on the input
  dependent_vars = list(filter(lambda i: i not in input_vars, range(L ** 2)))
  assert len(dependent_vars) == L ** 2 // 2
  for i in dependent_vars:
    row = J[i]
    # the variable depends on itself in the realnvp transform
    row[i] = 0
    arr = row.reshape((L, L))
    recovered_mask = (arr != 0).astype(np.float32)
    print('reconstructed mask for variable ', i)
    print(recovered_mask)
    assert np.array_equal(recovered_mask, np_mask) or np.array_equal(recovered_mask, 1 - np_mask)


def test_fast_jacobian(L):
  coupling = fast_coupling()
  _test_jacobian(coupling, L)


def test_perm_jacobian(L):
  coupling = perm_coupling(L)
  _test_jacobian(coupling, L)


if __name__ == '__main__':
  L = 1024
  test_posterior()
  test_prior()
  #_test_speeds(L, num_blocks=2)
  # coupling = perm_coupling(L, parity=True)
  # _test_coupling(coupling, L)
  # _test_jacobian(coupling, L)
  #coupling = fast_coupling()
  #_test_jacobian(coupling, 4)
  #_test_jacobian(coupling, 4)

  #test_fast(4)
  # num_blocks = 5
  # for L in [512, 1024]:
  #   print('\n--------------\nL = ', L)
  #   # print('\nconv:')
  #   # _test_memory_conv(L, num_blocks)
  #   print('\nfast, with roll')
  #   _test_memory_fast(L, num_blocks)
  #   torch.cuda.reset_max_memory_allocated()

  #   print('\npermutations')
  #   _test_memory_perm(L, num_blocks)
  #   torch.cuda.reset_max_memory_allocated()
  #test_conv2d_jacobian()
  #test_fast_coupling()
  #test_fast_coupling()
  #test_fast_jacobian()
