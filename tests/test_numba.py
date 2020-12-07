import sys, os
import numpy as np
import time
import numba
import scipy.sparse

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from adgm import utils

def simplex_projection(c):
    """
    return a solution to: min ||x - c||_2^2 s.t. dot(1, x) = 1 and x >= 0
    """
    n = len(c)
    a = -np.sort(-c)
    lambdas = (np.cumsum(a) - 1)/np.arange(1, n+1)
    for k in range(n-1, -1, -1):
        if a[k] > lambdas[k]:
            return np.maximum(c - lambdas[k], 0)


def simplex_projection_rowwise(C):
    """
    Doing simplex projection for each row of C
    """
    X = np.zeros(C.shape)
    for i in range(C.shape[0]):
        X[i] = simplex_projection(C[i])
    return X


@numba.jit(nopython=True, parallel=True, cache=True)
def simplex_projection_rowwise_numba(C):
    """
    Doing simplex projection for each row of C
    TODO: simplex_projection for performance
    """
    m = C.shape[0]
    n = C.shape[1]
    X = np.zeros((m, n))
    for i in numba.prange(m):
        a = -np.sort(-C[i])
        lambdas = (np.cumsum(a) - 1)/np.arange(1, n+1)
        for k in range(n-1, -1, -1):
            if a[k] > lambdas[k]:
                X[i] = np.maximum(C[i] - lambdas[k], 0)
                break
    return X


def benchmark(m=500, n=500, repeat=5):
    C = np.random.randn(m, n)

    for i in range(repeat):
        # numpy version
        start = time.time()
        X1 = simplex_projection_rowwise(C)
        numpy_time = time.time() - start

        # numba version
        start = time.time()
        X2 = simplex_projection_rowwise_numba(C)
        numba_time = time.time() - start
        print('{}) numpy: {:6f}, numba: {:6f}'.format(i + 1, numpy_time, numba_time))


def test_simplex_projection():
    A = scipy.sparse.random(4, 5, density=0.5, format='csr')
    # x = simplex_projection_inequality_sparse2(A.data, A.indices, A.indptr, A.shape[0])
    x = utils.simplex_projection_inequality_sparse(A.data, A.indices, A.indptr, A.shape[0])
    print(x)


@numba.jit(nopython=True)
def test(a):
    v = np.zeros(a.shape[0])


if __name__ == "__main__":
    # benchmark()
    # test(np.zeros(3))
    test_simplex_projection()
