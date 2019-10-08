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
import time
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix, csr_matrix
import numba


def build_pairwise_vectorized(points1, points2, len_weight=0.5):
    """
    points1: n1 x 2 array
    points2: n2 x 2 array
    For each pair of matches: i <--> p and j <--> q, the pairwise cost
        P(ip, jq) = w*leng + (1 - w)*0.5*(1 - cos_angle)
    where 
        leng = np.abs(norm1 - norm2)/(norm1 + norm2)
        cos_angle = ((xi-xj)*(xp-xq) + (yi-yj)*(yp-yq))/(norm1*norm2)
        norm1 = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))
        norm2 = np.sqrt((xp-xq)*(xp-xq) + (yp-yq)*(yp-yq))
        w = len_weight
    """
    len_weight = min(1.0, max(len_weight, 0))
    n1 = len(points1)
    n2 = len(points2)
    N = n1*n2

    x_left = points1[:, 0]
    y_left = points1[:, 1]
    x_right = points2[:, 0]
    y_right = points2[:, 1]

    # UX(i, j) = x_i - x_j
    # UX = np.repeat(np.transpose([x_left]), n1, axis=1) - np.repeat([x_left], n1, axis=0)
    UY = np.repeat(np.transpose([y_left]), n1, axis=1) - np.repeat([y_left], n1, axis=0)
    VX = np.repeat(np.transpose([x_right]), n2, axis=1) - np.repeat([x_right], n2, axis=0)
    VY = np.repeat(np.transpose([y_right]), n2, axis=1) - np.repeat([y_right], n2, axis=0)

    UX = np.outer(x_left, np.ones(n1))
    UX -= np.transpose(UX)
    UY = np.outer(y_left, np.ones(n1))
    UY -= np.transpose(UY)
    VX = np.outer(x_right, np.ones(n2))
    VX -= np.transpose(VX)
    VY = np.outer(y_right, np.ones(n2))
    VY -= np.transpose(VY)

    N1 = n1*n1
    N2 = n2*n2

    # print('UX.shape =', UX.shape)
    # print('VX.shape =', VX.shape)

    # M = len_weight*A + (1-len_weight)*0.5*(1 - B)
    UX = np.reshape(UX, N1)
    UY = np.reshape(UY, N1)
    VX = np.reshape(VX, N2)
    VY = np.reshape(VY, N2)

    # print('UX.shape =', UX.shape)
    # print('VX.shape =', VX.shape)

    normU = np.sqrt(UX**2 + UY**2)
    normV = np.sqrt(VX**2 + VY**2)

    # print('normU.shape =', normU.shape)
    # print('normV.shape =', normV.shape)

    # normU_rep = np.repeat(np.transpose([normU]), N2, axis=1)
    # normV_rep = np.repeat([normV], N1, axis=0)
    normU_rep = np.outer(normU, np.ones(N2))
    normV_rep = np.outer(np.ones(N1), normV)

    # print('normU_rep.shape =', normU_rep.shape)
    # print('normV_rep.shape =', normV_rep.shape)

    non_zeros_indices = (normU_rep > 0) & (normV_rep > 0)

    if np.count_nonzero(non_zeros_indices) == 0:
        M = np.zeros((N, N))
    else:
        A = np.abs(normU_rep - normV_rep)
        A = np.divide(A, normU_rep + normV_rep, out=np.zeros_like(A), where=non_zeros_indices)

        # UX_rep = np.repeat(np.transpose([UX]), N2, axis=1)
        # UY_rep = np.repeat(np.transpose([UY]), N2, axis=1)
        # VX_rep = np.repeat([VX], N1, axis=0)
        # VY_rep = np.repeat([VY], N1, axis=0)
        # B = UX_rep*VX_rep + UY_rep*VY_rep

        B = np.outer(UX, VX) + np.outer(UY, VY)

        B = np.divide(B, normU_rep*normV_rep, out=np.zeros_like(B), where=non_zeros_indices)
        M = len_weight*A + ((1-len_weight)*0.5)*(1 - B)
        M[~non_zeros_indices] = 0
        # print('M =', M)
        M = np.reshape(np.swapaxes(np.reshape(M, (n1, n1, n2, n2)), 1, 2), (N, N))
        # print('M =', M)

    # print("building pairwise potentials took: ", time.time() - start)

    # # DEBUG: compare with naive solution
    # # build pairwise potentials
    # # print('Building pairwise potentials...')
    # start = time.time()
    # affinityMatrix = np.zeros((N, N))
    # for a in range(N - 1):
    #     p = a % n2
    #     i = int((a - p)/n2)
    #     pt_i = objects1[i]
    #     pd_p = objects2[p]
    #     xi, yi = pt_i.x, pt_i.y
    #     xp, yp = pd_p.x, pd_p.y
    #     for b in range(a + 1, N):
    #         q = b % n2
    #         j = int((b - q)/n2)
    #         if q == p or j == i:
    #             continue
    #         pt_j = objects1[j]
    #         pd_q = objects2[q]
    #         xj, yj = pt_j.x, pt_j.y
    #         xq, yq = pd_q.x, pd_q.y

    #         norm1 = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))
    #         norm2 = np.sqrt((xp-xq)*(xp-xq) + (yp-yq)*(yp-yq))
    #         leng = np.abs(norm1 - norm2)/(norm1 + norm2)
    #         cos_angle = ((xi-xj)*(xp-xq) + (yi-yj)*(yp-yq))/(norm1*norm2)
    #         tt = len_weight*leng + (1 - len_weight)*0.5*(1 - cos_angle)
    #         affinityMatrix[a, b] = tt
    #         affinityMatrix[b, a] = tt
    
    # # print('M naive =', affinityMatrix)

    # print("naive solution took: ", time.time() - start)

    # print('norm(vectorized - naive) =', np.linalg.norm(M - affinityMatrix))

    return M



@numba.jit(nopython=True)
def pairwise_cost(edge1, edge2, len_weight=1.0):
    """
    pairwise cost when matchin i <--> p, j <-->q
    edge1: [xi, yi, xj, yj]
    edge2: [xp, yp, xq, yq]
    NOTE: it's much faster to not use auxiliary functions inside numba.jit
        so the purpose of this function is just for testing.
    """
    xi, yi, xj, yj = edge1
    xp, yp, xq, yq = edge2
    norm1 = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))
    norm2 = np.sqrt((xp-xq)*(xp-xq) + (yp-yq)*(yp-yq))
    leng = np.abs(norm1 - norm2)/(norm1 + norm2)
    cos_angle = ((xi-xj)*(xp-xq) + (yi-yj)*(yp-yq))/(norm1*norm2)
    tt = len_weight*leng + (1 - len_weight)*0.5*(1 - cos_angle)
    return tt



@numba.jit(nopython=True, parallel=True, cache=True)
def build_pairwise_dense(points1, points2, len_weight=0.5):
    """
    points1: n1 x 2 array
    points2: n2 x 2 array
    For each pair of matches: i <--> p and j <--> q, the pairwise cost
        P(ip, jq) = w*leng + (1 - w)*0.5*(1 - cos_angle)
    where 
        leng = np.abs(norm1 - norm2)/(norm1 + norm2)
        cos_angle = ((xi-xj)*(xp-xq) + (yi-yj)*(yp-yq))/(norm1*norm2)
        norm1 = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))
        norm2 = np.sqrt((xp-xq)*(xp-xq) + (yp-yq)*(yp-yq))
        w = len_weight
    """
    len_weight = min(1.0, max(len_weight, 0))
    n1 = len(points1)
    n2 = len(points2)
    N = n1*n2
    
    # lidx, ridx = np.unravel_index(range(N), (n1, n2))

    P = np.zeros((N, N))
    for a in numba.prange(N - 1):
        p = a % n2
        i = int((a - p)/n2)
        # p = ridx[a]
        # i = lidx[a]
        xi = points1[i, 0]
        yi = points1[i, 1]
        xp = points2[p, 0]
        yp = points2[p, 1]
        for b in range(a + 1, N):
            q = b % n2
            j = int((b - q)/n2)
            # q = ridx[b]
            # j = lidx[b]
            if q == p or j == i:
                continue
            xj = points1[j, 0]
            yj = points1[j, 1]
            xq = points2[q, 0]
            yq = points2[q, 1]

            # tt = pairwise_cost([xi, yi, xj, yj], [xp, yp, xq, yq], len_weight=len_weight)

            norm1 = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))
            norm2 = np.sqrt((xp-xq)*(xp-xq) + (yp-yq)*(yp-yq))
            leng = np.abs(norm1 - norm2)/(norm1 + norm2)
            cos_angle = ((xi-xj)*(xp-xq) + (yi-yj)*(yp-yq))/(norm1*norm2)
            tt = len_weight*leng + (1 - len_weight)*0.5*(1 - cos_angle)
            
            P[a, b] = tt
            P[b, a] = tt

    return P


# @numba.jit('Tuple((f8[:], f8[:]))(i4[:,:], i4[:,:], optional(i4[:,:]), optional(i4[:,:]), optional(f8[:,:]), optional(f8))', nopython=True, parallel=True)
@numba.jit(nopython=True, parallel=True, cache=True)
def toto(points1, points2, len_weight, row, col):
    nnz = row.shape[0]
    n1 = len(points1)
    n2 = len(points2)
    data = np.zeros(nnz)
    for idx in numba.prange(nnz):
        a = row[idx]
        b = col[idx]
        p = a % n2
        i = int((a - p)/n2)
        q = b % n2
        j = int((b - q)/n2)
        # print('i = {}, j = {}, p = {}, q = {}'.format(i, j, p, q))
        xi = points1[i, 0]
        yi = points1[i, 1]
        xj = points1[j, 0]
        yj = points1[j, 1]
        xp = points2[p, 0]
        yp = points2[p, 1]
        xq = points2[q, 0]
        yq = points2[q, 1]

        norm1 = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))
        norm2 = np.sqrt((xp-xq)*(xp-xq) + (yp-yq)*(yp-yq))
        leng = np.abs(norm1 - norm2)/(norm1 + norm2)
        cos_angle = ((xi-xj)*(xp-xq) + (yi-yj)*(yp-yq))/(norm1*norm2)
        tt = len_weight*leng + (1 - len_weight)*0.5*(1 - cos_angle)
        data[idx] = tt
    return data



def build_pairwise_sparse(points1, points2, adj1=None, adj2=None, assignment_mask=None, len_weight=0.5):
    """
    points1: n1 x 2 array
    points2: n2 x 2 array
    For each pair of matches: i <--> p and j <--> q, the pairwise cost
        P(ip, jq) = w*leng + (1 - w)*0.5*(1 - cos_angle)
    where 
        leng = np.abs(norm1 - norm2)/(norm1 + norm2)
        cos_angle = ((xi-xj)*(xp-xq) + (yi-yj)*(yp-yq))/(norm1*norm2)
        norm1 = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))
        norm2 = np.sqrt((xp-xq)*(xp-xq) + (yp-yq)*(yp-yq))
        w = len_weight
    adj1, adj2: n1xn1, n2xn2 adjacency matrices of points1 and points2.
        If None then fully connected.
    assignment_mask: the mask for the assignment matrix
        assignment_mask[i, p] = 0 means i should never match to p
    """
    len_weight = min(1.0, max(len_weight, 0))
    n1 = len(points1)
    n2 = len(points2)
    N = n1*n2
    
    # all assignment indices
    if assignment_mask is None:
        assignment_mask = np.ones((n1, n2))
    aidx = np.flatnonzero(assignment_mask)
    A = len(aidx)

    start = time.time()

    # pre-compute x,y coordinates of each linear index 
    # lidx, ridx = np.unravel_index(range(N), (n1, n2))
    lidx = np.zeros(A)
    ridx = np.zeros(A)
    for idx_a in range(A):
        a = aidx[idx_a]
        p = a % n2
        i = int((a - p)/n2)
        lidx[idx_a] = i
        ridx[idx_a] = p
    
    print('pre-processing 0:', time.time() - start)

    start = time.time()
    # # count the number of non-zero element in the final pairwise matrix
    # # not easy to get a formula, so we can iterate, but it's slow!
    # # upper-bound for the total number of non-zeros: min(E1*E2, A*(A-1)/2)
    # nnz = 0
    # for idx_a in range(A - 1):
    #     i = lidx[idx_a]
    #     p = ridx[idx_a]
    #     for idx_b in range(idx_a + 1, A):
    #         j = lidx[idx_b]
    #         q = ridx[idx_b]
    #         if q == p or j == i:
    #             continue
    #         # if ij or pq is not an edge then no potential
    #         if adj1 is not None and adj1[i, j] <= 0:
    #             continue
    #         if adj2 is not None and adj2[p, q] <= 0:
    #             continue
    #         nnz += 1

    # print('pre-processing 1:', time.time() - start)

    # start = time.time()

    # # now we iterate again to get the indices i, j, p, q
    # # last column of data is the potential
    # row = np.zeros(nnz, dtype=np.int)
    # col = np.zeros(nnz, dtype=np.int)
    # idx = 0
    # for idx_a in range(A - 1):
    #     a = aidx[idx_a]
    #     i = lidx[idx_a]
    #     p = ridx[idx_a]
    #     for idx_b in range(idx_a + 1, A):
    #         b = aidx[idx_b]
    #         j = lidx[idx_b]
    #         q = ridx[idx_b]
    #         if q == p or j == i:
    #             continue
    #         # if ij or pq is not an edge then no potential
    #         if adj1 is not None and adj1[i, j] <= 0:
    #             continue
    #         if adj2 is not None and adj2[p, q] <= 0:
    #             continue
    #         row[idx] = a
    #         col[idx] = b
    #         idx += 1


    row = []
    col = []
    for idx_a in range(A - 1):
        a = aidx[idx_a]
        i = lidx[idx_a]
        p = ridx[idx_a]
        for idx_b in range(idx_a + 1, A):
            b = aidx[idx_b]
            j = lidx[idx_b]
            q = ridx[idx_b]
            if q == p or j == i:
                continue
            # if ij or pq is not an edge then no potential
            # if adj1 is not None and adj1[i, j] <= 0:
            #     continue
            # if adj2 is not None and adj2[p, q] <= 0:
            #     continue
            # row.append(a)
            # col.append(b)

    row = np.asarray(row)
    col = np.asarray(col)

    print('pre-processing 2:', time.time() - start)

    # start = time.time()
    # data = toto(points1, points2, len_weight, row, col)
    data = np.zeros(1)
    # print('computing:', time.time() - start)
    
    # start = time.time()
    # making the potential matrix symmetric before returning
    data = np.tile(data, 2)
    row_ = np.concatenate((row, col))
    col_ = np.concatenate((col, row))
    # print('post-processing:', time.time() - start)

    return data, row_, col_, (N, N)


# @numba.jit('Tuple((float64[:], float64[:,:]))(float64[:], float64[:,:], optional(float64[:,:]))',nopython=True)
# def f(a, b, c=None) :
#     return a, b


@numba.jit(nopython=True, parallel=True, cache=True)
def tata(points1, points2, AIDX, len_weight):
    n1 = len(points1)
    n2 = len(points2)
    A = len(AIDX)
    P = np.zeros((A, A))
    for ia in numba.prange(A - 1):
        a = AIDX[ia]
        p = a % n2
        i = int((a - p)/n2)
        # p = ridx[a]
        # i = lidx[a]
        xi = points1[i, 0]
        yi = points1[i, 1]
        xp = points2[p, 0]
        yp = points2[p, 1]
        for ib in range(ia + 1, A):
            b = AIDX[ib]
            q = b % n2
            j = int((b - q)/n2)
            # q = ridx[b]
            # j = lidx[b]
            if q == p or j == i:
                continue
            xj = points1[j, 0]
            yj = points1[j, 1]
            xq = points2[q, 0]
            yq = points2[q, 1]

            norm1 = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))
            norm2 = np.sqrt((xp-xq)*(xp-xq) + (yp-yq)*(yp-yq))
            leng = np.abs(norm1 - norm2)/(norm1 + norm2)
            cos_angle = ((xi-xj)*(xp-xq) + (yi-yj)*(yp-yq))/(norm1*norm2)
            tt = len_weight*leng + (1 - len_weight)*0.5*(1 - cos_angle)
            
            P[ia, ib] = tt
            P[ib, ia] = tt
    return P


def build_pairwise_sparse_assignment(points1, points2, assignment_mask=None, len_weight=0.5):
    """
    sparse assignment, dense graphs
    """
    len_weight = min(1.0, max(len_weight, 0))
    n1 = len(points1)
    n2 = len(points2)
    
    # if the assignment is not provided then it's assumed to be dense
    if assignment_mask is None:
        assignment_mask = np.ones((n1, n2))
    
    # list of all assignment (linear) indices
    AIDX = np.flatnonzero(assignment_mask)
    # compute the potential matrix
    P = tata(points1, points2, AIDX, len_weight)
    return P


def match_points(points1, points2, unaryCost=None, graph_type='full'):
    """

    """
    # for i in range(3):
    #     start = time.time()
    #     P = build_pairwise_dense(points1, points2, len_weight=1.0)
    #     vectorized_time = time.time() - start

    #     start = time.time()
    #     P2 = build_pairwise_dense_numba(points1, points2, len_weight=1.0)
    #     numba_time = time.time() - start
    #     print("{}) vectorized = {}, numba = {}".format(i + 1, vectorized_time, numba_time))
    #     # print('norm = ', np.linalg.norm(P - P2))
    
    start = time.time()
    P = build_pairwise_dense_numba(points1, points2, len_weight=1.0)
    print("Dense pairwise potentials took:", time.time() - start)

    start = time.time()
    data, row, col, shape = build_pairwise_sparse(points1, points2, len_weight=1.0)
    print("Sparse pairwise potentials took:", time.time() - start)

    coo = coo_matrix((data, (row, col)), shape=shape)
    P_sparse = coo.todense()

    print('norm =', np.linalg.norm(P - P_sparse))


    # for i in range(3):
    #     start = time.time()
    #     P = build_pairwise_dense(points1, points2, len_weight=1.0)
    #     vectorized_time = time.time() - start

    #     start = time.time()
    #     P2 = build_pairwise_dense_numba(points1, points2, len_weight=1.0)
    #     numba_time = time.time() - start
    #     print("{}) vectorized = {}, numba = {}".format(i + 1, vectorized_time, numba_time))
    #     # print('norm = ', np.linalg.norm(P - P2))

    # start = time.time()
    # U = np.zeros((len(points1), len(points2)))
    # X = ADGM(U, P, X0=None, verbose=True, max_iter=100)
    # print("ADGM took: ", time.time() - start)

    return X


def matching(features1, features2):
    """
    features1: n1 x 2 array
    features2: n2 x 2 array
    """
    tri = Delaunay(features1)
    