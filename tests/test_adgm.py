import sys, os
import numpy as np
import time
import numba
import random
import time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from adgm.adgm import ADGM_D

np.random.seed(12)
random.seed(12)

def test_ADGM_D():
    n1 = 10
    n2 = 15

    U = np.random.rand(n1, n2)

    # random adjacency matrices
    adj1 = np.random.rand(n1, n1) >= 0.5
    np.fill_diagonal(adj1, 0)
    adj2 = np.random.rand(n2, n2) >= 0.5
    np.fill_diagonal(adj2, 0)
    edges1 = np.transpose(np.stack(np.nonzero(adj1)))
    edges2 = np.transpose(np.stack(np.nonzero(adj2)))
    Q = np.random.rand(len(edges1), len(edges2))
    print(f'Q shape = {Q.shape}')

    kwargs = {'scheme': 'fixed', 'rho': 1000.0, 'max_iter': 100,
              'projection': 'euclidean',
              'verbose': True}

    start = time.time()
    ADGM_D(U, Q, edges1, edges2, **kwargs)
    print(f'Time = {time.time() - start}')


if __name__ == "__main__":
    test_ADGM_D()
