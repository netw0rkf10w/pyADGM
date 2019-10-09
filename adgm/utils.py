"""
Alternating Direction Graph Matching

Copyright @ D. Khue Le-Huu
https://khue.fr

If you use any part of this code, please cite the following paper.
 
D. Khuê Lê-Huu and Nikos Paragios. Alternating Direction Graph Matching.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

BibTeX:

@inproceedings{lehuu2017adgm,
  title={Alternating Direction Graph Matching},
  author={L{\^e}-Huu, D. Khu{\^e} and Paragios, Nikos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}

This file is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY, without even the implied 
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
import numpy as np
import numba

import matplotlib.pyplot as plt
from matplotlib import collections  as mc

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


def simplex_projection_inequality(c):
    """
    return a solution to: min ||x - c||_2^2 s.t. dot(1, x) <= 1 and x >= 0
    """
    u = np.maximum(c, 0)
    if np.sum(u) <= 1:
        return u
    return simplex_projection(c)


def simplex_projection_rowwise_(C):
    """
    Doing simplex projection for each row of C
    TODO: simplex_projection for performance
    """
    X = np.zeros(C.shape)
    for i in range(C.shape[0]):
        X[i] = simplex_projection(C[i])
    return X

def simplex_projection_inequality_rowwise_(C):
    """
    Doing simplex projection for each row of C
    TODO: simplex_projection for performance
    """
    X = np.zeros(C.shape)
    for i in range(C.shape[0]):
        X[i] = simplex_projection_inequality(C[i])
    return X


@numba.jit(nopython=True, parallel=True, cache=True)
def simplex_projection_rowwise(C):
    """
    Doing simplex projection for each row of C
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


@numba.jit(nopython=True, parallel=True, cache=True)
def simplex_projection_inequality_rowwise(C):
    """
    Doing simplex projection for each row of C
    """
    m = C.shape[0]
    n = C.shape[1]
    X = np.zeros((m, n))
    for i in numba.prange(m):
        u = np.maximum(C[i], 0)
        if np.sum(u) <= 1:
            X[i] = u
        else:
            a = -np.sort(-C[i])
            lambdas = (np.cumsum(a) - 1)/np.arange(1, n+1)
            for k in range(n-1, -1, -1):
                if a[k] > lambdas[k]:
                    X[i] = np.maximum(C[i] - lambdas[k], 0)
                    break
    return X


@numba.jit(nopython=True, parallel=True, cache=True)
def simplex_projection_sparse(data, indices, indptr, num_vectors):
    """
    simplex project of each row or each column of a sparse matrix C
    If C is CSR: each row; if C is CSC: each column
    data, indices, indptr, shape: representation of C (same notation as scipy csr/csc matrix)
    num_vectors = number of rows if C is CSR
                = number of cols if C is CSC
    """
    x = np.zeros(len(data))
    for i in numba.prange(num_vectors):
        # projection for each vector independently
        start = indptr[i]
        end = indptr[i+1]
        if end <= start:
            continue
        ci = data[start:end]
        ni = end - start
        a = -np.sort(-ci)
        lambdas = (np.cumsum(a) - 1)/np.arange(1, ni+1)  
        for k in range(ni-1, -1, -1):
            if a[k] > lambdas[k]:
                xi = np.maximum(ci - lambdas[k], 0)
                break
        x[start:end] = xi
    return x


@numba.jit(nopython=True, parallel=True, cache=True)
def simplex_projection_inequality_sparse(data, indices, indptr, num_vectors):
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
        if end <= start:
            continue
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


@numba.jit(nopython=True, parallel=True, cache=True)
def matvec_fast(x, Adata, Aindices, Aindptr, Ashape):
    """
    Fast sparse matrix-vector multiplication
    https://stackoverflow.com/a/47830250/2131200
    Note: the first call of this function will be slow
        because numba needs to initialize.
    """
    m = Ashape[0]    
    Ax = np.zeros(numRowsA)

    for i in numba.prange(m):
        Ax_i = 0.0        
        for idx in range(Aindptr[i], Aindptr[i+1]):
            j = Aindices[idx]
            Ax_i += Adata[idx]*x[j]
        Ax[i] = Ax_i
    return Ax


def matvec(x, Adata, Aindices, Aindptr, Ashape):
    """
    naive
    """
    numRowsA = Ashape[0]    
    Ax = np.zeros(numRowsA)

    for i in range(numRowsA):
        Ax_i = 0.0        
        for dataIdx in range(Aindptr[i], Aindptr[i+1]):
            j = Aindices[dataIdx]
            Ax_i += Adata[dataIdx]*x[j]

        Ax[i] = Ax_i
    return Ax


@numba.jit(nopython=True, parallel=True, cache=True)
def pairwise_distance(points1, points2):
    """
    Fast sparse matrix-vector multiplication
    https://stackoverflow.com/a/47830250/2131200
    Note: the first call of this function will be slow
        because numba needs to initialize.
    """
    n1 = len(points2)
    n2 = len(points2)
    N = n1*n2

    # store the indices
    data = np.zeros((N, 3))
    for a in range(N):
        j = a % n2
        i = int((a - j)/n2)
        data[a, 0] = i
        data[a, 1] = j

    for a in numba.prange(N):
        i = int(data[a, 0])
        j = int(data[a, 1])
        xi = points1[i, 0]
        yi = points1[i, 1]
        xj = points2[j, 0]
        yj = points2[j, 1]
        # compute the distance and store in the last column
        data[a, 2] = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))

    return data


# @numba.jit(nopython=True, parallel=True, cache=True)
def reshape_sparse(x, x_linear_indices, x_dim, n1, n2):
    """
    reshape x to a n1 x n2 sparse matrix X
    x_linear_indices represent the linear indices of
        each element of x in X
    """
    X = np.zeros((n1, n2))
    for ia in range(x_dim):
        a = x_linear_indices[ia]
        p = a % n2
        i = int((a - p)/n2)
        print('ia = {}, i = {}, p = {}'.format(ia, i, p))
        X[i, p] = x[ia]
    return X

def draw_matches(plot, points1, points2, matches=None,
                 color1='b', color2='b', colorm='g', s=50, linewidth=2):
    fig, ax = plot
    # Draw point based on above x, y axis values.
    plt.scatter(points1[:, 0], points1[:, 1], s=s)
    plt.scatter(points2[:, 0], points2[:, 1], s=s)
    if matches is not None:
        lines = []
        for i1, i2 in matches:
            lines.append([points1[i1], points2[i2]])

        colors = [colorm]*len(lines)
        lc = mc.LineCollection(lines, colors=colors, linewidths=linewidth)
        ax.add_collection(lc)


def matches_from_assignment_matrix(X):
    """
    return a list of matches
    """
    return np.transpose(np.nonzero(X))