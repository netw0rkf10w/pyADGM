import os
import numpy as np
import random
import time

from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

from adgm.adgm import ADGM
from adgm.energy import build_pairwise_potentials
from utils import draw_results, get_adjacency_matrix

# for reproducibility
np.random.seed(12)
random.seed(12)

def main():
    # numbers of points
    n1 = 40
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

    # Add random potential assignments
    # assignment_mask = np.logical_or(np.random.randn(n1, n2) > 0.5, X_gt)
    assignment_mask = None

    # where to save outputs
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    # Visualize the feature points and the ground-truth matching
    plot = plt.subplots()
    plot[1].title.set_text('Ground-truth matching')
    draw_results(plot, points1, points2, X=X_gt)
    plt.savefig(os.path.join(output_dir, 'ground-truth.jpg'), dpi=600, bbox_inches='tight')
    
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
    U = np.zeros((n1, n2))

    # Build the pairwise potentials with fully-connected graphs
    start = time.time()
    P_dense = build_pairwise_potentials(points1, points2, 
                        assignment_mask=assignment_mask, len_weight=len_weight)
    print('Building dense potentials time (s):', time.time() - start)
    # Call ADGM solver
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
    start = time.time()
    X_dense = ADGM(U, P_dense, assignment_mask=assignment_mask, **kwargs)
    print('Fully-connected matching time (s):', time.time() - start)
    # Plot the results
    plot = plt.subplots()
    plot[1].title.set_text('Fully-connected graph matching')
    draw_results(plot, points1, points2, X=X_dense, X_gt=X_gt)
    plt.savefig(os.path.join(output_dir, 'dense.jpg'), dpi=600, bbox_inches='tight')
    
    # Sparse graphs
    # Building the graphs based on Delaunay triangulation
    tri1 = Delaunay(points1)
    adj1 = get_adjacency_matrix(tri1)
    tri2 = Delaunay(points2)
    adj2 = get_adjacency_matrix(tri2)
    start = time.time()
    # Build the pairwise potentials with Delaunay graphs
    P_sparse = build_pairwise_potentials(points1, points2,
                        assignment_mask=assignment_mask, adj1=adj1, adj2=adj2,
                        len_weight=len_weight)
    print('Building sparse potentials time (s):', time.time() - start)
    # Call ADGM solver
    start = time.time()
    X_sparse = ADGM(U, P_sparse, assignment_mask=assignment_mask, **kwargs)
    print('Sparse matching time (s):', time.time() - start)
    # Plot the results
    plot = plt.subplots()
    plot[1].title.set_text('Sparse graph matching')
    plt.triplot(points1[:,0], points1[:,1], tri1.simplices, color='b')
    plt.triplot(points2[:,0], points2[:,1], tri2.simplices, color='b')
    draw_results(plot, points1, points2, X=X_dense, X_gt=X_gt)
    plt.savefig(os.path.join(output_dir, 'sparse.jpg'), dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()