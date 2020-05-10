import math


class Decay(object):
  def __init__(self, start_value, end_value, decay_rate, decay_steps):
    self.start_value = start_value
    self.end_value = end_value
    self.decay_rate = decay_rate
    self.decay_steps = decay_steps
    self.check_end_value()

  def print_values(self, num_values):
    print('--------------------')
    print('printing value of decay over time:')
    for step in range(0, self.decay_steps * 2, int(self.decay_steps // num_values)):
      value = self.get_value(step)
      print(f'step {step}:\tvalue: {value:.3e}')

  def get_value(self, step):
    return max(self.end_value, self._get_value(step))
  
  def check_end_value(self):
    end_value = self.get_value(self.decay_steps)
    if end_value > self.end_value:
      raise ValueError(f'decay_steps: {self.decay_steps}'
                       f'\tTarget end_value: {self.end_value}'
                       f'\tActual end_value: {end_value}')

  def _get_value(self, step):
    raise NotImplementedError


class NaturalExpDecay(Decay):

  def _get_value(self, step):
    return (self.start_value * 
            math.exp(-self.decay_rate * step / self.decay_steps))


class ExponentialDecay(Decay):
  
  def _get_value(self, step):
    if step >= self.decay_steps:
      return 0
    else:
      return self.start_value * self.decay_rate ** (step / self.decay_steps)


class InverseTimeDecay(Decay):
  
  def _get_value(self, step):
    return (self.start_value / 
            (1 + self.decay_rate * step / self.decay_steps))


class PolynomialDecay(Decay):
  
  def _get_value(self, step):
    return ((self.start_value - self.end_value) *
            (1 - step / self.decay_steps) ** self.decay_rate
            + self.end_value)

  
