import torch
from torch import nn
from torch.nn import init
from .scalar_linear import ElementwiseLinear

class ScalarSingleNeuronBasicBlock(nn.Module):
  """Batched scalar input and output, with single-neuron hidden layers."""
  def __init__(self, in_features):
    super().__init__()
    self.linear1 = ElementwiseLinear(in_features, out_features=1)
    self.bn1 = nn.BatchNorm1d(in_features)
    self.relu = nn.ReLU(inplace=True)
    self.linear2 = ElementwiseLinear(in_features, out_features=1)
    self.bn2 = nn.BatchNorm1d(in_features)

  def forward(self, x):
    identity = x
    out = self.linear1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.linear2(out)
    out = self.bn2(out)

    out += identity
    out = self.relu(out)

    return out
    

class ScalarSingleNeuronResNet(nn.Module):
  """Residual neural network with single neurons per hidden layer."""
  def __init__(self, block, layer_num_blocks, input_size, outputs_per_input):
    super().__init__()
    self.layer1 = self._make_layer(block, input_size, layer_num_blocks[0])
    self.layer2 = self._make_layer(block, input_size, layer_num_blocks[1])
    self.layer3 = self._make_layer(block, input_size, layer_num_blocks[2])
    self.layer4 = self._make_layer(block, input_size, layer_num_blocks[3])
    self.linear = ElementwiseLinear(input_size, outputs_per_input)

    for m in self.modules():
      if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, ElementwiseLinear):
        m.reset_parameters()
  
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    for m in self.modules():   
      if isinstance(m, ScalarSingleNeuronBasicBlock):
        nn.init.constant_(m.bn2.weight, 0)

  def _make_layer(self, block, in_features, blocks):
    layers = []
    for _ in range(blocks):
      layers.append(block(in_features))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    # x is of shape (batch_size, input_size)
    # linear layer weight, bias are of shape (input_size, outputs_per_input)
    # so need to unsqueeze x so it is broadcastable
    return self.linear(x.unsqueeze(-1))


