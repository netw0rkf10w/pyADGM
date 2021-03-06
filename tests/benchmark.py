import sys, os
import numpy as np
import scipy.sparse
import time
import random

from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from adgm.adgm import ADGM, ADGM_D, rounding
from adgm.energy import build_pairwise_potentials, build_Q_edges
from utils import draw_results, get_adjacency_matrix

# np.random.seed(12)
# random.seed(12)


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


def benchmark_sparse():
    # numbers of points
    n1 = 30
    n2 = 30
    # in this example, we randomly take some points of the first set
    # then randomly transform them, thus we need n1 >= n2
    assert n1 >= n2

    # create a set of randoms 2D points, zero-centered them
    points1 = np.random.randint(100, size=(n1, 2))
    points1 = points1 - np.mean(points1, axis=0)

    # randomly transform it using a similarity transformation

    # first construct the transformation matrix
    theta = np.random.rand()*np.pi/2
    # scale = np.random.uniform(low=0.5, high=1.5)
    scale = 0.9
    tx = np.random.randint(low=120, high=150)
    ty = np.random.randint(50)
    M = np.array([[scale*np.cos(theta), np.sin(theta),       tx],
                    [-np.sin(theta),      scale*np.cos(theta), ty]])

    # then transform the first set of points
    points2 = np.ones((3, n1))
    points2[:2] = np.transpose(points1)
    points2 = np.transpose(np.dot(M, points2))

    # randomly keep only n2 points
    indices = list(range(n1))
    random.shuffle(indices)
    points2 = points2[indices]
    points2 = points2[:n2]

    # ground-truth matching, for evaluation
    X_gt = np.zeros((n1, n2), dtype=int)
    for idx2, idx1 in enumerate(indices[:n2]):
        X_gt[idx1, idx2] = 1

    # where to save outputs
    # output_dir = './output'
    # os.makedirs(output_dir, exist_ok=True)

    # Visualize the feature points and the ground-truth matching
    plot = plt.subplots()
    plot[1].title.set_text('Ground-truth matching')
    draw_results(plot, points1, points2, X=X_gt)
    # plt.savefig(os.path.join(output_dir, 'ground-truth.jpg'), dpi=600, bbox_inches='tight')
    
    # Graph matching

    # Weight of length with respect to angle
    # If the scales of the the sets of points are roughly the same then this
    # value should be high (max = 1.0)
    # If the scales are very different but the poses are roughly the same 
    # (i.e. small rotation) then this value should be small (min = 0.0)
    # If both scales and poses are very different, then using sparse graphs
    # (e.g., Delaunay triangulation) may help, but for difficult cases
    # one may have to use higher-order potentials.
    len_weight = 0.7

    # We do not use any unary potentials here
    # U = np.zeros((n1, n2))
    U = np.random.rand(n1, n2)
    
    # Call ADGM solver
    # ADGM parameters
    kwargs = {'rho': max(10**(-60.0/np.sqrt(n1*n2)), 1e-4),
              'rho_max': 100,
              'step': 1.2,
              'precision': 1e-5,
              'decrease_delta': 1e-3,
              'iter1': 5,
              'iter2': 10,
              'max_iter': 10000,
              'verbose': False}
    
    # Sparse graphs
    # Building the graphs based on Delaunay triangulation
    tri1 = Delaunay(points1)
    adj1 = get_adjacency_matrix(tri1)
    tri2 = Delaunay(points2)
    adj2 = get_adjacency_matrix(tri2)


    # Build the pairwise potentials with fully-connected graphs
    start = time.time()
    Q, edges1, edges2 = build_Q_edges(points1, points2, adj1, adj2, len_weight=len_weight)
    print('Building Q edges time (s):', time.time() - start)

    start = time.time()
    # kwargs.update({'scheme': 'fixed', 'max_iter': 100, 'projection': 'euclidean'})
    X = ADGM_D(U, Q, edges1, edges2, **kwargs)
    print('ADGM_D time (s):', time.time() - start)
    # Plot the results
    plot = plt.subplots()
    plot[1].title.set_text('ADGM_D')
    plt.triplot(points1[:,0], points1[:,1], tri1.simplices, color='b')
    plt.triplot(points2[:,0], points2[:,1], tri2.simplices, color='b')
    draw_results(plot, points1, points2, X=X, X_gt=X_gt)
    # plt.savefig(os.path.join(output_dir, 'dense.jpg'), dpi=600, bbox_inches='tight')

    start = time.time()
    # Build the pairwise potentials with Delaunay graphs
    P_sparse = build_pairwise_potentials(points1, points2, adj1=adj1, adj2=adj2,
                        len_weight=len_weight)
    print('Building sparse potentials time (s):', time.time() - start)
    # Call ADGM solver
    start = time.time()
    X_sparse = ADGM(U, P_sparse, **kwargs)
    print('ADGM time (s):', time.time() - start)
    print(f'norm(X - X_sparse) = {np.linalg.norm(X.astype(float) - X_sparse.astype(float))}')
    # Plot the results
    plot = plt.subplots()
    plot[1].title.set_text('ADGM')
    plt.triplot(points1[:,0], points1[:,1], tri1.simplices, color='b')
    plt.triplot(points2[:,0], points2[:,1], tri2.simplices, color='b')
    draw_results(plot, points1, points2, X=X_sparse, X_gt=X_gt)
    # plt.savefig(os.path.join(output_dir, 'sparse.jpg'), dpi=600, bbox_inches='tight')


    start = time.time()
    kwargs.update({'projection': 'bregman', 'verbose': False})
    X = ADGM_D(U, Q, edges1, edges2, **kwargs)
    print('ADGM_D bregman time (s):', time.time() - start)
    # Plot the results
    plot = plt.subplots()
    plot[1].title.set_text('ADGM_D Bregman')
    plt.triplot(points1[:,0], points1[:,1], tri1.simplices, color='b')
    plt.triplot(points2[:,0], points2[:,1], tri2.simplices, color='b')
    draw_results(plot, points1, points2, X=X, X_gt=X_gt)

    start = time.time()
    kwargs.update({'scheme': 'fixed', 'max_iter': 100, 'projection': 'bregman'})
    X = ADGM_D(U, Q, edges1, edges2, **kwargs)
    print('ADGM_D fixed bregman time (s):', time.time() - start)
    X = rounding(X)
    # Plot the results
    plot = plt.subplots()
    plot[1].title.set_text('ADGM_D Fixed Bregman')
    plt.triplot(points1[:,0], points1[:,1], tri1.simplices, color='b')
    plt.triplot(points2[:,0], points2[:,1], tri2.simplices, color='b')
    draw_results(plot, points1, points2, X=X, X_gt=X_gt)

    plt.show()
        


if __name__ == "__main__":
    # check_matvec(m=10000, n=10000, density=0.1, repeat=3)
    # test2()
    # check_potentials(n1=40, n2=30, repeat=5)
    benchmark_sparse()