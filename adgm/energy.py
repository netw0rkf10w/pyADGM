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


@numba.jit(nopython=True, parallel=True, cache=True)
def build_pairwise(points1, points2, AIDX, len_weight):
    n1 = len(points1)
    n2 = len(points2)
    A = len(AIDX)
    P = np.zeros((A, A))
    for ia in numba.prange(A - 1):
        a = AIDX[ia]
        p = a % n2
        i = int((a - p)/n2)
        xi = points1[i, 0]
        yi = points1[i, 1]
        xp = points2[p, 0]
        yp = points2[p, 1]
        for jb in range(ia + 1, A):
            b = AIDX[jb]
            q = b % n2
            j = int((b - q)/n2)
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
            tt = len_weight*(leng - 1) + (1 - len_weight)*0.5*(-cos_angle - 1)
            
            P[ia, jb] = tt
            P[jb, ia] = tt
    return P


@numba.jit(nopython=True, parallel=True, cache=True)
def build_pairwise_adj(points1, points2, AIDX, len_weight, adj1, ajd2):
    n1 = len(points1)
    n2 = len(points2)
    A = len(AIDX)
    P = np.zeros((A, A))
    for ia in numba.prange(A - 1):
        a = AIDX[ia]
        p = a % n2
        i = int((a - p)/n2)
        xi = points1[i, 0]
        yi = points1[i, 1]
        xp = points2[p, 0]
        yp = points2[p, 1]
        for jb in range(ia + 1, A):
            b = AIDX[jb]
            q = b % n2
            j = int((b - q)/n2)
            if q == p or j == i:
                continue
            if adj1[i, j] <= 0 or ajd2[p, q] <= 0:
                continue
            xj = points1[j, 0]
            yj = points1[j, 1]
            xq = points2[q, 0]
            yq = points2[q, 1]

            norm1 = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))
            norm2 = np.sqrt((xp-xq)*(xp-xq) + (yp-yq)*(yp-yq))
            leng = np.abs(norm1 - norm2)/(norm1 + norm2)
            cos_angle = ((xi-xj)*(xp-xq) + (yi-yj)*(yp-yq))/(norm1*norm2)
            tt = len_weight*(leng - 1) + (1 - len_weight)*0.5*(-cos_angle - 1)
            
            P[ia, jb] = tt
            P[jb, ia] = tt
    return P


def build_pairwise_potentials(points1, points2, assignment_mask=None, adj1=None, adj2=None, len_weight=0.5):
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
    if adj1 is None and adj2 is None:
        P = build_pairwise(points1, points2, AIDX, len_weight)
    else:
        if adj1 is None:
            adj1 = np.ones((n1, n1), dtype=np.int64)
        elif adj2 is None:
            adj2 = np.ones((n2, n2), dtype=np.int64)
        P = build_pairwise_adj(points1, points2, AIDX, len_weight, adj1, adj2)
    return P
