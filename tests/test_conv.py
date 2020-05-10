import torch
import network


def test_checkerboard_shapes():
  net = network.Conv2dCheckerboard(inner_activation='leaky_relu', final_activation=None, hidden_size=16)
  for L in [4, 10, 16, 32]:
    x = torch.round(torch.rand(1, L, L)) * 2 - 1
    out = net(x)
    print('L', L, 'out:', out.shape)
    # shifting the odd rows of the checkerboard makes half columns black, half white
    # so the output we expect is the same number of rows, but half the columns
    # stride is (1, 2)
    assert tuple(out.shape) == (1, L, L // 2)



def test_rectangle_shapes():
  net = network.Conv2dRectangle(inner_activation='leaky_relu',
                                final_activation=None,
                                hidden_size=16)
  for L in [4, 10, 16, 32]:
    x = torch.round(torch.rand(1, L, L // 2)) * 2 - 1
    out = net(x)
    print('L', L, 'out:', out.shape)
    assert tuple(out.shape) == (1, L, L // 2)


if __name__ == '__main__':
  #test_checkerboard_shapes()
  test_rectangle_shapes()
