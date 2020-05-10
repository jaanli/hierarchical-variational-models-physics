import numpy as np
import torch

L = 4
idx = np.arange(L ** 2).reshape((L, L))

checker_idx = idx.copy()
checker_idx[1::2] = np.roll(idx[1::2], -1, axis=-1)
print('original')
print(idx)
print('checker')
print(checker_idx)

# even columns
# NB: ravel works as in pytorch, 'C' row-major, C-style order, last axis index changing fastest, first axis index changing slowest
even_idx = checker_idx[..., ::2].ravel()
odd_idx = checker_idx[..., 1::2].ravel()

# permuted indices with even cols are first, odd cols last
permuted_idx = np.concatenate((even_idx, odd_idx))

# can we sort the flattened original lattice of indices to get this permutation
print('permuted index from original index so even/odd are first half/last half')
permuted = idx.ravel()[permuted_idx]
print(permuted)
print('permuted index with even/odd in first/last half')
print(permuted_idx)

# can we get back to checker_idx from the even/odd halves?
unperm = np.argsort(permuted_idx)
orig_ravel = permuted[unperm]
print('unpermuted, original')
print(orig_ravel)
print(idx.ravel())

print('reshaped, original idx')
print(orig_ravel.reshape((L, L)))
print(idx)


# try in pytorch
num_samples = 3
idx = torch.arange(L ** 2).reshape((L, L))
idx = idx.unsqueeze(0).repeat((num_samples, 1, 1))
checker_idx = idx.clone()
checker_idx[..., 1::2, :] = idx[..., 1::2, :].roll(-1, dims=-1)
print(checker_idx)
permuted_idx = idx.view(idx.shape[0], -1)[..., permuted_idx]
even, odd = permuted_idx.chunk(2, dim=-1)
print('half of checker board indices')
print(even.view((even.shape[0], L, L // 2)))
print('opposite color of checkerboard indices')
print(odd.view((odd.shape[0], L, L // 2)))
