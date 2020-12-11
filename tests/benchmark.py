import sys, os
import numpy as np
import scipy.sparse
import time
import random

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
# from adgm.adgm import *
from adgm.utils import matvec_fast, matvec
# from adgm.energy import *

np.random.seed(12)
random.seed(12)


def check_matvec(m=10000, n=10000, density=0.1, repeat=5):
    for i in range(repeat):
        # m += 100*i
        # n += 100*i
        Acoo = scipy.sparse.random(m, n, density=density, format='coo')
        A = Acoo.tocsr()
        x = np.random.randn(A.shape[1])
        # x = scipy.sparse.random(m, n, density=density, format='csr')

        start = time.time()
        Ax = A.dot(x)
        scipy_time = time.time() - start

        start = time.time()
        AxCheck = matvec_fast(x, A.data, A.indices, A.indptr, A.shape)
        matvec_fast_time = time.time() - start

        start = time.time()
        AxCheck2 = matvec(x, Acoo.data, Acoo.row, Acoo.col, Acoo.shape)
        matvec_time = time.time() - start

        # print('norm = ', np.linalg.norm(AxCheck - AxCheck2))

        # if m < 10000 and n < 10000:
        #     # numpy dense array
        #     A_np = A.todense()
        #     start = time.time()
        #     Ax_np = A_np.dot(x)
        #     numpy_time = time.time() - start
        #     # print('norm(scipy - numpy) =', np.linalg.norm(Ax - Ax_np))
        #     start = time.time()
        #     Ax_naive = matvec(x, A.data, A.indices, A.indptr, A.shape)
        #     naive_time = time.time() - start

        #     print('{}) scipy: {:6f} (s), matvec: {:6f} (s), numpy: {:6f}, naive: {:6f}'.format(i + 1, scipy_time, matvec_time, numpy_time, naive_time))
        # else:
        #     print('{}) scipy: {:6f} (s), matvec: {:6f} (s)'.format(i + 1, scipy_time, matvec_time))

        # start = time.time()
        # # Axmv = matvec.matvec(x, A.data, A.indices, A.indptr, A.shape[0])
        # precompiled_time = time.time() - start

        print('{}) scipy: {:6f} (s), matvec_fast: {:6f} (s), matvec: {:6f}'.format(i + 1, scipy_time, matvec_fast_time, matvec_time))
        # print('norm(scipy - matvec) = {}, norm(scipy - precompiled) = {}'.format(np.linalg.norm(Ax - AxCheck), np.linalg.norm(Ax - Axmv)))

        # time.sleep(2)

def test():
    A = scipy.sparse.random(5, 5, density=0.5, format='csr')
    x = np.random.randn(A.shape[1])
    Axmv = matvec.matvec(x, A.data, A.indices, A.indptr, A.shape[0])
    print(Axmv)


def test2():
    m = 10000
    n = 10000
    for i in range(5):
        print(i)
        A = np.random.rand(m, n)
        x = np.random.randn(A.shape[1])
        Ax = A.dot(x)
        time.sleep(2)


def check_potentials(n1, n2, repeat=5):
    assert n1 >= n2

    # create a set of randoms 2D points, zero-centered them
    points1 = np.random.randint(100, size=(n1, 2))
    points1 = points1 - np.mean(points1, axis=0)

    # randomly transform it
    theta = np.random.rand()*np.pi/2
    # scale = np.random.uniform(low=0.5, high=1.5)
    scale = 1
    tx = np.random.randint(low=120, high=150)
    ty = np.random.randint(50)
    M = np.array([[scale*np.cos(theta), np.sin(theta),       tx],
                  [-np.sin(theta),       scale*np.cos(theta),  ty]])
    # transform the first set of features
    points2 = np.ones((3, n1))
    points2[:2] = np.transpose(points1)
    points2 = np.transpose(np.dot(M, points2))
    
    # randomly choose n2 points
    indices = list(range(n1))
    random.shuffle(indices)
    points2 = points2[indices]
    points2 = points2[:n2]

    # ground-truth matching
    X_gt = np.zeros((n1, n2), dtype=int)
    for idx2, idx1 in enumerate(indices[:n2]):
        X_gt[idx1, idx2] = 1

    # Add random potential assignments
    assignment_mask = np.logical_or(np.random.randn(n1, n2) > 0.5, X_gt)

    for i in range(repeat):
        start = time.time()
        # P = build_pairwise_dense(points1, points2)
        vectorized_time = time.time() - start

        start = time.time()
        P2 = build_pairwise_dense_numba(points1, points2)
        numba_time = time.time() - start

        start = time.time()
        # data, row, col, shape = build_pairwise_sparse(points1, points2)
        P3 = build_pairwise_sparse_assignment(points1, points2, assignment_mask=assignment_mask)
        sparse_time = time.time() - start

        # print("{}) vectorized = {}, numba = {}, sparse = {}".format(i + 1, vectorized_time, numba_time, sparse_time))

        # P3 = reshape_sparse(data, AIDX, len(AIDX), n1, n2)
        # print('norm(P2 - P3) = {}'.format(np.linalg.norm(P2 - P3)))
        # print(P)
        # print(data)

        # coo = coo_matrix((data, (row, col)), shape=shape)
        # P_sparse = coo.todense()
        # print('norm(vect - numb) = {}, norm(vect - spar) = {}'.format(np.linalg.norm(P - P2), np.linalg.norm(P - P_sparse)))

        max_iter = 5
        U = np.zeros((len(points1), len(points2)))
        verbose = False

        # Dense ADGM
        start = time.time()
        X = ADGM(U, P2, X0=None, verbose=verbose, max_iter=max_iter)
        dense_time = time.time() - start

        start = time.time()
        X = ADGM_sparse_assignment(U, P3, assignment_mask=assignment_mask, X0=None, verbose=verbose, max_iter=max_iter)
        sparse_time = time.time() - start

        print("{}) adgm_dense = {}, adgm_sparse = {}".format(i + 1, dense_time, sparse_time))
        


if __name__ == "__main__":
    check_matvec(m=10000, n=10000, density=0.1, repeat=3)
    # test2()
    # check_potentials(n1=40, n2=30, repeat=5)