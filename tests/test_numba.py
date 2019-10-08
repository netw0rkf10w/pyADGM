import numpy as np
import time
import numba
import scipy.sparse

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


@numba.jit(nopython=True, parallel=True, cache=True)
def simplex_projection_inequality_sparse2(data, indices, indptr, num_vectors):
    """
    simplex project (with inequality constraints) of each row or each column of a sparse matrix C
    If C is CSR: each row; if C is CSC: each column
    data, indices, indptr, shape: representation of C (same notation as scipy csr/csc matrix)
    num_vectors = number of rows if C is CSR
                = number of cols if C is CSC
    """
    x = np.zeros(len(data))
    for i in numba.prange(num_vectors):
        # projection for each row independently
        start = indptr[i]
        end = indptr[i+1]
        ci = data[start:end]
        u = np.maximum(ci, 0)
        if np.sum(u) <= 1:
            xi = u
        else:
            ni = end - start
            a = -np.sort(-ci)
            lambdas = (np.cumsum(a) - 1)/np.arange(1, ni+1)  
            for k in range(ni-1, -1, -1):
                if a[k] > lambdas[k]:
                    xi = np.maximum(ci - lambdas[k], 0)
                    break
        x[start:end] = xi
    return x


def test_simplex_projection():
    A = scipy.sparse.random(4, 5, density=0.5, format='csr')
    x = simplex_projection_inequality_sparse2(A.data, A.indices, A.indptr, A.shape[0])
    print(x)



@numba.jit(nopython=True)
def test(a):
    v = np.zeros(a.shape[0])


if __name__ == "__main__":
    # benchmark()
    # test(np.zeros(3))
    test_simplex_projection()
