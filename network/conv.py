import torch
from torch import nn

ACTIVATION = {'leaky_relu': lambda: nn.LeakyReLU(inplace=True), 
              'tanh': nn.Tanh}


class Conv2d(nn.Module):
    def __init__(self, hidden_sizes,
                 use_final_bias, inner_activation, final_activation, use_dropout,
                 use_weight_norm, kernel_size):
        output_shape = (None, None)
        input_shape = output_shape
        assert(isinstance(input_shape, tuple))
        assert(isinstance(use_final_bias, bool))
        assert(len(input_shape) == 2 or len(input_shape) == 3)
        assert(len(output_shape) == 2 or len(output_shape) == 3)
        super().__init__()
        if len(input_shape) == 2:
            input_shape = tuple([1] + list(input_shape))
        input_channels = input_shape[0]
        output_channels = 1 if len(output_shape) == 2 else output_shape[0]
        self.input_shape = input_shape
        self.output_shape = output_shape
        sizes = [input_channels] + hidden_sizes + [output_channels]
        layers = []
        padding_size = kernel_size-1 # padding size in circ mode is weird?
        for i in range(len(sizes) - 1):
          layers.append(nn.Conv2d(sizes[i], sizes[i+1], kernel_size,
                                  padding=padding_size, stride=1,
                                  padding_mode='circular'))
          if use_weight_norm:
            layers[-1] = torch.nn.utils.weight_norm(layers[-1])
          if i < len(sizes)-2: # mid layers
            layers.append(ACTIVATION[inner_activation]())
            if use_dropout: layers.append(torch.nn.Dropout(p=use_dropout))
        if final_activation is not None:
            layers.append(ACTIVATION[final_activation]())
        self.net = nn.Sequential(*layers)

    # add channels -> net -> remove channels
    def forward(self, x, context=None):
        shape = x.shape
        x = x.unsqueeze(1) # channel dim
        out = self.net(x)
        return out.view(shape).squeeze(1) 


class Conv2dDilatedCheckerboard(nn.Module):
  """Implicit checkerboard masking through dilated, strided convolution.

  See figure 3 in https://arxiv.org/pdf/1605.08803.pdf
  
  This operation reduces the input dimensions (black squares) by half.

  The outputs correspond to white squares.

  Assumes:
    - every odd dimension of input is rolled to the left (zero-indexed).
    - inputs of shape (L, L) where L is an even number
    - the output of this op will be used to transform the white squares

  Args:
    - hidden_size: number of dimensions for every kernel parameter
  """
  def __init__(self, hidden_size, inner_activation, final_activation):
    super().__init__()
    modules = []
    modules.append(nn.Conv2d(in_channels=1,
                             out_channels=hidden_size,
                             kernel_size=2,
                             stride=(1, 2),
                             padding=(1, 1),
                             dilation=(2, 2)))
    modules.append(ACTIVATION[inner_activation]())
    modules.append(nn.Conv2d(in_channels=hidden_size,
                             out_channels=1,
                             kernel_size=(1, 1),
                             stride=1,
                             padding=0,
                             dilation=1))
    if final_activation is not None:
      modules.append(ACTIVATION[final_activation]())
    self.net = nn.Sequential(*modules)

  def forward(self, input, context=None):
    """Input is of shape (num_samples, L, L) where L is lattice length.

    Output: (num_samples, L, L // 2)
    """
    input = input.unsqueeze(1)  # unsqueeze in_channels dimension
    return self.net(input).squeeze(1) # squeeze the out_channels dim



class Conv2dRect(nn.Module):
  """Conv2d operating on a (L, L // 2) shaped rectangle."""
  def __init__(self, in_channels, hidden_size, inner_activation, final_activation):
    super().__init__()
    modules = []
    # pad the left and top by one to get 'SAME' (as input size) 
    # output
    #modules.append(nn.ReplicationPad2d((1, 0, 1, 0)))
    #modules.append(nn.ZeroPad2d((1, 0, 1, 0)))
    modules.append(nn.Conv2d(in_channels=in_channels,
                             out_channels=hidden_size,
                             kernel_size=3,
                             stride=(1, 1),
                             padding=2,
                             dilation=1,
                             padding_mode='circular'))
    modules.append(ACTIVATION[inner_activation]())
    modules.append(nn.Conv2d(in_channels=hidden_size,
                             out_channels=1,
                             kernel_size=3,
                             stride=1,
                             padding=(1, 1),
                             dilation=1))
    if final_activation is not None:
      modules.append(ACTIVATION[final_activation]())
    self.net = nn.Sequential(*modules)

  def forward(self, input, context=None):
    """Input is of shape (num_samples, L, L) where L is lattice length.

    Output: (num_samples, L, L // 2)
    """
    if context is not None:
      input = torch.stack([input] + context, dim=1)
    if input.ndim  == 3:
      input = input.unsqueeze(1)  # unsqueeze in_channels dimension
    return self.net(input).squeeze(1) # squeeze the out_channels dim
