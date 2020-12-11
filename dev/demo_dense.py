import numpy as np
import random
import time

import matplotlib.pyplot as plt

from adgm.adgm import ADGM, ADGM_sparse_assignment
from adgm.energy import build_pairwise_dense, build_pairwise_sparse_assignment
from adgm.utils import draw_matches, matches_from_assignment_matrix

np.random.seed(12)
random.seed(12)

def main():
    n1 = 40
    n2 = 30
    assert n1 >= n2

    # create a set of randoms 2D points, zero-centered them
    points1 = np.random.randint(100, size=(n1, 2))
    points1 = points1 - np.mean(points1, axis=0)

    # randomly transform it
    theta = np.random.rand()*np.pi/2
    # scale = np.random.uniform(low=0.5, high=1.5)
    scale = 0.9
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
    # assignment_mask = np.logical_or(np.random.randn(n1, n2) > 0.5, X_gt)
    assignment_mask = None

    # We will try a dense version and a sparse version of ADGM to see the difference in performance

    # First define some parameters
    # ADGM parameters
    kwargs = {'rho_min': max(10**(-60.0/np.sqrt(n1*n2)), 1e-4),
              'rho_max': 100,
              'step': 1.2,
              'precision': 1e-5,
              'decrease_delta': 1e-3,
              'iter1': 5,
              'iter2': 10,
              'max_iter': 10000,
              'verbose': False}
    

    # Energy parameters
    # weight of length with respect to angle
    # If the scales of the the sets of points are roughly the same then this value should be high (max = 1.0)
    # If the scales are very different but the poses are roughly the same (i.e. small rotation) then this value
    # should be small (min = 0.0)
    # There scales are different and the rotation is large, then we would need higher-order potentials
    len_weight = 0.7 

    # We do not use any unary potentials here
    U = np.zeros((len(points1), len(points2)))

    # Dense version
    start = time.time()
    P_dense = build_pairwise_dense(points1, points2, len_weight=len_weight)
    print('Building potentials time (s):', time.time() - start)
    X_dense = ADGM(U, P_dense, **kwargs)
    print('Dense matching time (s):', time.time() - start)

    # Plot the matches
    plot = plt.subplots()
    plot[1].title.set_text('Dense')
    draw_matches(plot, points1, points2, matches=matches_from_assignment_matrix(X_dense), colorm='r')
    # Draw the ground-truth matches in green
    draw_matches(plot, points1, points2, matches=matches_from_assignment_matrix(X_gt))
    # plt.show()

    # Sparse version
    start = time.time()
    P_sparse = build_pairwise_sparse_assignment(points1, points2, assignment_mask=assignment_mask, len_weight=len_weight)
    print('Building potentials time (s):', time.time() - start)
    X_sparse = ADGM_sparse_assignment(U, P_sparse, assignment_mask=assignment_mask, **kwargs)
    print('Sparse matching time (s):', time.time() - start)

    # Plot the matches
    plot = plt.subplots()
    plot[1].title.set_text('Sparse')
    draw_matches(plot, points1, points2, matches=matches_from_assignment_matrix(X_sparse), colorm='r')
    # Draw the ground-truth matches in green
    draw_matches(plot, points1, points2, matches=matches_from_assignment_matrix(X_gt))
    plt.show()

if __name__ == "__main__":
    main()