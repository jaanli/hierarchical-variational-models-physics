import matplotlib
matplotlib.use('Agg')

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_style('white')
sns.set_context('talk')


def plot_heatmap(mat, fname):
  """Plot covariance of N samples of dimension D, shape (D, N)."""
  system_size = mat.shape[0]
  fig, ax = plt.subplots(figsize=(5 * 1.618, 5))
  sns.heatmap(mat[::-1], ax=ax, xticklabels=[], yticklabels=[])
  plt.savefig(fname, bbox_inches='tight')
  plt.close()


def plot_sample(sample, fname):
  """Black and white plot of Ising model samples."""
  fig, ax = plt.subplots(figsize=(5, 5))
  ax.imshow(sample, cmap=plt.cm.gray)
  ax.set_xticks([])
  ax.set_yticks([])
  plt.savefig(fname, bbox_inches='tight')
  plt.close()
