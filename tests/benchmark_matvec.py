import sys, os
import numpy as np
import scipy.sparse
import time
import random

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from adgm.lemmas import matvec, matvec_csr, matvec_coo

# np.random.seed(12)
# random.seed(12)

def test_matvec_coo():
    m = 10
    n = 5
    A = scipy.sparse.random(m, n, density=0.5, format='coo')
    x = np.random.randn(A.shape[1])
    y = matvec_coo(x, A.data, A.row, A.col, A.shape)

def test_matvec_csr():
    m = 10
    n = 5
    A = scipy.sparse.random(m, n, density=0.5, format='coo')
    A = A.tocsr()
    x = np.random.randn(A.shape[1])
    y = matvec_csr(x, A.data, A.indices, A.indptr, A.shape)

def benchmark_matvec(exp=5, density=0.25, repeat=10):
    """Benchmark for dimensions 10, 100, 1000,...,10^exp
    """
    # For caching
    test_matvec_coo()
    test_matvec_csr()

    dims = [10**(a+1) for a in range(exp)]
    for dim in dims:
        scipy_time = 0.0
        matvec_coo_time = 0.0
        matvec_csr_time = 0.0
        matvec_csrs_time = 0.0
        for i in range(repeat):
            Acoo = scipy.sparse.random(dim, dim, density=density, format='coo')
            A = Acoo.tocsr()
            x = np.random.randn(A.shape[1])
            # x = scipy.sparse.random(m, n, density=density, format='csr')

            start = time.time()
            y = A.dot(x)
            scipy_time += time.time() - start

            y2 =  Acoo.dot(x)

            start = time.time()
            y_coo = matvec_coo(x, Acoo.data, Acoo.row, Acoo.col, Acoo.shape)
            matvec_coo_time += time.time() - start

            norm_coo = np.linalg.norm(y2 - y_coo)
            # print(f'norm_coo = {norm_coo}')
            # assert norm_coo < 1e-10

            start = time.time()
            y_csr = matvec_csr(x, A.data, A.indices, A.indptr, A.shape)
            matvec_csr_time += time.time() - start

            start = time.time()
            y_csrs = matvec(x, A.data, A.indices, A.indptr, A.shape)
            matvec_csrs_time += time.time() - start

            # norm_csr = np.linalg.norm(y - y_csr)
            # print(f'norm_csr = {norm_csr}')
            # assert norm_csr < 1e-10
            
        print(f'dim {dim} -- scipy: {scipy_time:6f} (s), '
              f'matvec_coo: {matvec_coo_time:6f} (s), '
              f'matvec_csr: {matvec_csr_time:6f} (s), '
              f'matvec_csrs: {matvec_csrs_time:6f} (s)')


if __name__ == "__main__":
    benchmark_matvec(exp=3, density=0.5, repeat=5)
    # test_matvec_coo()