import pathlib
import os

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def add_bool(parser, arg, help=''):
  parser.add_argument(arg,
                      type=str2bool,
                      nargs='?',
                      const=True,
                      default=False,
                      help=help)


def expand_path(string):
  return pathlib.Path(os.path.expandvars(string))


def args_to_string(args):
  res = {}
  for k, v in args.__dict__.items():
    if isinstance(v, pathlib.Path):
      v = os.fspath(v)
    res[k] = v
  return res


def add_model(parser):
  parser.add_argument('--model',
                      choices=['ising', 'sk'],
                      help='Model name')
  parser.add_argument('--beta',
                      type=float,
                      help='Thermodynamic inverse temperature')
  parser.add_argument('--num_spins',
                      type=int,
                      help='Total number of spins in model.')
  parser.add_argument('--boundary',
                      choices=['periodic', 'open'],
                      help='Boundary for Ising model')
                      
  
def add_variational(parser):
  parser.add_argument('--variational_posterior',
                      choices=['BinaryPosterior', 'ConditionalPosterior', 'BinaryConditionalPosterior', 'RealNVPPosterior'],
                      help='Variational distribution')
  parser.add_argument('--hidden_size',
                      type=int,
                      help='Hidden size')
  parser.add_argument('--activation',
                      choices=['relu', 'leaky_relu', 'tanh'],
                      help='Activation function for neural nets')
  parser.add_argument('--flow_type',
                      choices=['autoregressive', 'realnvp', 'realnvp_full_conditioning'],
                      help='Number of flow blocks to use')
  parser.add_argument('--flow_depth',
                      type=int,
                      help='Number of flow blocks to use')
  parser.add_argument('--prior_std',
                      type=float,
                      help='Standard deviation of prior base density')
  parser.add_argument('--posterior_std',
                      type=float,
                      help='Standard deviation of posterior base density')
  add_bool(parser, '--initialize_flow_std', 'Initialize flow scale')
  add_bool(parser, '--reverse', 'Reverse input order every flow block')
  parser.add_argument('--hidden_degrees',
                      choices=['random', 'equal'],
                      help='How to assign hidden node degrees in MADE')



def add_optim(parser):
  add_bool(parser, '--use_gpu', 'Use GPU')
  parser.add_argument('--seed',
                      type=int,
                      help='Random seed')
  parser.add_argument('--checkpoint',
                      type=expand_path,
                      help='Path to saved parameter dictionary')
  parser.add_argument('--learning_rate',
                      type=float,
                      help='Learning rate')
  parser.add_argument('--momentum',
                      type=float,
                      help='Momentum')
  parser.add_argument('--num_samples_grad',
                      type=int,
                      help='Number of samples for estimating gradient')
  parser.add_argument('--max_iteration',
                      type=int,
                      help='Maximum iterations')
  add_bool(parser, '--rao_blackwellize', 'Use Rao-Blackwellization')
  add_bool(parser, '--marginalize', 'Marginalize latent variables')
  add_bool(parser, 
           '--control_variate',
           'Use score function as control variate with optimal scaling')


def add_annealing(parser):
  parser.add_argument('--annealing_decay',
                      choices=['PolynomialDecay', 
                               'NaturalExpDecay',
                               'ExponentialDecay',
                               'InverseTimeDecay'],
                      default=None,
                      help='Type of decay to use for annealing')
  parser.add_argument('--annealing_init_scale',
                      type=float,
                      help='Initial scale for annealing temperature')
  parser.add_argument('--decay_steps',
                      type=int,
                      help='Steps to anneal over')
  parser.add_argument('--decay_rate',
                      type=float,
                      help='Rate for annealing')


def add_proximity(parser):
  add_bool(parser, '--proximity_constraint', 'Use proximity variational inference')
  parser.add_argument('--moving_average_decay',
                      type=float,
                      help='Decay for moving average')
  parser.add_argument('--init_magnitude_scale',
                      type=float,
                      help='Initial scale for proximity constraint')
  parser.add_argument('--decay_steps',
                      type=int,
                      help='Steps to anneal over')
  parser.add_argument('--decay_rate',
                      type=float,
                      help='Exponential decay rate for magnitude of proximity constraint')
  

def add_logging(parser):
  parser.add_argument('--num_samples_print',
                      type=int,
                      help='Number of samples to draw for printing')
  parser.add_argument('--print_batch_size',
                      type=int,
                      help='Batch size for printing')
  parser.add_argument('--log_interval',
                      type=int,
                      help='Number of steps between logging')
  parser.add_argument('--log_dir',
                      type=expand_path,
                      help='Path for writing log files')
